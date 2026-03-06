// End-to-end SSD: 0.5B draft (ANE) + 3B target (GPU/ANE)
// Real different-sized models, real acceptance logic.

import Foundation
import CoreML

@main
struct SSDE2E {
    static let maxTokens = 128
    static let draftCount = 2

    static func main() async throws {
        let draftPath = NSString(string: "~/models/qwen2.5-0.5b-coreml/Qwen2.5-0.5B-Instruct-4bit.mlmodelc").expandingTildeInPath
        let targetPath = NSString(string: "~/models/qwen2.5-3b-coreml/Qwen2.5-3B-Instruct-4bit.mlmodelc").expandingTildeInPath

        print("Loading 3B target...")
        let targetConfig = MLModelConfiguration()
        targetConfig.computeUnits = .all
        let targetModel = try MLModel(contentsOf: URL(fileURLWithPath: targetPath), configuration: targetConfig)

        print("Loading 0.5B draft...")
        let draftConfig = MLModelConfiguration()
        draftConfig.computeUnits = .all
        let draftModel = try MLModel(contentsOf: URL(fileURLWithPath: draftPath), configuration: draftConfig)

        print("Both loaded. draft=0.5B, target=3B, maxTokens=\(maxTokens), draftCount=\(draftCount)")
        print(String(repeating: "=", count: 60))

        // Warmup
        let warmDraft = draftModel.makeState()
        let warmTarget = targetModel.makeState()
        let warmInput = try makeInput(token: 42)
        for _ in 0..<5 {
            _ = try await draftModel.prediction(from: warmInput, using: warmDraft)
            _ = try await targetModel.prediction(from: warmInput, using: warmTarget)
        }

        // === 1. VANILLA: 3B target only ===
        print("\n1. Vanilla (3B target only)")
        let vState = targetModel.makeState()
        var tok: Int32 = 42
        let vStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<maxTokens {
            let r = try await targetModel.prediction(from: makeInput(token: tok), using: vState)
            tok = argmax(r.featureValue(for: "logits")!.multiArrayValue!)
        }
        let vTime = CFAbsoluteTimeGetCurrent() - vStart
        let vTps = Double(maxTokens) / vTime
        print(String(format: "   %d tokens in %.2fs = %.1f tok/s", maxTokens, vTime, vTps))

        // === 2. STANDARD SPEC DECODE: draft then verify (sequential) ===
        print("\n2. Standard Spec Decode (0.5B draft → 3B verify, sequential)")
        let (specToks, specTime, specAcc, specRounds) = try await runSpecDecode(
            draft: draftModel, target: targetModel, parallel: false)
        let specTps = Double(specToks) / specTime
        print(String(format: "   %d tokens in %.2fs = %.1f tok/s (accept=%.0f%%, rounds=%d)",
                     specToks, specTime, specTps, Double(specAcc)/Double(specToks)*100, specRounds))

        // === 3. SSD: draft overlaps with verify (parallel) ===
        print("\n3. SSD (0.5B draft || 3B verify, parallel)")
        let (ssdToks, ssdTime, ssdAcc, ssdRounds) = try await runSpecDecode(
            draft: draftModel, target: targetModel, parallel: true)
        let ssdTps = Double(ssdToks) / ssdTime
        print(String(format: "   %d tokens in %.2fs = %.1f tok/s (accept=%.0f%%, rounds=%d)",
                     ssdToks, ssdTime, ssdTps, Double(ssdAcc)/Double(ssdToks)*100, ssdRounds))

        // Summary
        print("\n" + String(repeating: "=", count: 60))
        print(String(format: "Vanilla:     %.1f tok/s", vTps))
        print(String(format: "Spec:        %.1f tok/s (%.2fx vs vanilla)", specTps, specTps/vTps))
        print(String(format: "SSD:         %.1f tok/s (%.2fx vs vanilla)", ssdTps, ssdTps/vTps))
        print(String(format: "SSD vs Spec: %.2fx", ssdTps/specTps))
    }

    static func runSpecDecode(
        draft: MLModel, target: MLModel, parallel: Bool
    ) async throws -> (tokens: Int, time: Double, accepted: Int, rounds: Int) {
        let dState = draft.makeState()
        let tState = target.makeState()
        var tok: Int32 = 42
        var generated = 0
        var accepted = 0
        var rounds = 0

        let start = CFAbsoluteTimeGetCurrent()

        while generated < maxTokens {
            rounds += 1
            let numDraft = min(draftCount, maxTokens - generated)

            // Draft phase: generate numDraft tokens with 0.5B
            var draftTokens: [Int32] = []
            var draftTok = tok
            for _ in 0..<numDraft {
                let r = try await draft.prediction(from: makeInput(token: draftTok), using: dState)
                draftTok = argmax(r.featureValue(for: "logits")!.multiArrayValue!)
                draftTokens.append(draftTok)
            }

            if parallel {
                // === SSD: start pre-speculating NEXT round on draft while verifying ===
                let nextStart = draftTokens.last!
                async let preSpec: [Int32] = {
                    var specToks: [Int32] = []
                    var sTok = nextStart
                    for _ in 0..<numDraft {
                        let r = try await draft.prediction(from: makeInput(token: sTok), using: dState)
                        sTok = argmax(r.featureValue(for: "logits")!.multiArrayValue!)
                        specToks.append(sTok)
                    }
                    return specToks
                }()

                // Verify on target (runs in parallel with pre-spec above)
                var verifyTok = tok
                var roundAccepted = 0
                for i in 0..<draftTokens.count {
                    let r = try await target.prediction(from: makeInput(token: verifyTok), using: tState)
                    let tTok = argmax(r.featureValue(for: "logits")!.multiArrayValue!)
                    if tTok == draftTokens[i] {
                        roundAccepted += 1
                        verifyTok = tTok
                        generated += 1
                        accepted += 1
                    } else {
                        verifyTok = tTok
                        generated += 1
                        break
                    }
                    if generated >= maxTokens { break }
                }

                if roundAccepted == draftTokens.count && generated < maxTokens {
                    let r = try await target.prediction(from: makeInput(token: verifyTok), using: tState)
                    verifyTok = argmax(r.featureValue(for: "logits")!.multiArrayValue!)
                    generated += 1
                }

                // Collect pre-spec (already done by now since target is slower)
                let _ = try await preSpec
                tok = verifyTok

            } else {
                // Sequential verify
                var verifyTok = tok
                for i in 0..<draftTokens.count {
                    let r = try await target.prediction(from: makeInput(token: verifyTok), using: tState)
                    let tTok = argmax(r.featureValue(for: "logits")!.multiArrayValue!)
                    if tTok == draftTokens[i] {
                        accepted += 1
                        verifyTok = tTok
                        generated += 1
                    } else {
                        verifyTok = tTok
                        generated += 1
                        break
                    }
                    if generated >= maxTokens { break }
                }

                if generated < maxTokens && verifyTok == draftTokens.last {
                    let r = try await target.prediction(from: makeInput(token: verifyTok), using: tState)
                    verifyTok = argmax(r.featureValue(for: "logits")!.multiArrayValue!)
                    generated += 1
                }

                tok = verifyTok
            }
        }

        return (generated, CFAbsoluteTimeGetCurrent() - start, accepted, rounds)
    }

    static func makeInput(token: Int32) throws -> MLDictionaryFeatureProvider {
        let ids = try MLMultiArray(shape: [1, 1], dataType: .int32)
        ids[0] = NSNumber(value: token)
        let mask = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
        mask[0] = 0
        return try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
            "causal_mask": MLFeatureValue(multiArray: mask),
        ])
    }

    static func argmax(_ arr: MLMultiArray) -> Int32 {
        let count = arr.count
        let ptr = arr.dataPointer.bindMemory(to: Float16.self, capacity: count)
        var maxIdx = 0
        var maxVal = ptr[0]
        for i in 1..<count {
            if ptr[i] > maxVal {
                maxVal = ptr[i]
                maxIdx = i
            }
        }
        return Int32(maxIdx)
    }
}
