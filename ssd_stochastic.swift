// SSD with stochastic acceptance: tolerates draft/target mismatch.
//
// Uses finnvoorhees 0.5B stateful model for BOTH draft and target
// but with DIFFERENT compute units (ANE vs GPU).
// Tokens diverge at greedy, but stochastic acceptance still works.
//
// Stochastic speculative decoding:
//   For each draft token x sampled at temperature T:
//     p_target = softmax(target_logits)[x]
//     p_draft = softmax(draft_logits)[x]
//     if p_target >= p_draft: accept
//     else: accept with probability p_target / p_draft

import Foundation
import CoreML

@main
struct SSDStochastic {
    static let maxTokens = 128
    static let draftCount = 2
    static let temperature: Float = 0.8

    static func main() async throws {
        let draftPath = NSString(string: "~/models/qwen2.5-0.5b-coreml/Qwen2.5-0.5B-Instruct-4bit.mlmodelc").expandingTildeInPath
        let targetPath = NSString(string: "~/models/qwen2.5-3b-coreml/Qwen2.5-3B-Instruct-4bit.mlmodelc").expandingTildeInPath

        print("Loading models...")
        let targetConfig = MLModelConfiguration()
        targetConfig.computeUnits = .all
        let targetModel = try MLModel(contentsOf: URL(fileURLWithPath: targetPath), configuration: targetConfig)

        let draftConfig = MLModelConfiguration()
        draftConfig.computeUnits = .all
        let draftModel = try MLModel(contentsOf: URL(fileURLWithPath: draftPath), configuration: draftConfig)

        print("Draft: 0.5B, Target: 3B")

        print("Config: maxTokens=\(maxTokens) draft=\(draftCount) temp=\(temperature)")
        print(String(repeating: "=", count: 60))

        // Warmup
        let ws = targetModel.makeState()
        let wi = try makeInput(token: 42)
        for _ in 0..<5 { _ = try await targetModel.prediction(from: wi, using: ws) }

        // === 1. VANILLA ===
        print("\n1. Vanilla")
        let vState = targetModel.makeState()
        var tok: Int32 = 42
        let vStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<maxTokens {
            let r = try await targetModel.prediction(from: makeInput(token: tok), using: vState)
            tok = sampleTemp(r.featureValue(for: "logits")!.multiArrayValue!, temp: temperature)
        }
        let vTime = CFAbsoluteTimeGetCurrent() - vStart
        let vTps = Double(maxTokens) / vTime
        print(String(format: "   %.1f tok/s", vTps))

        // === 2. SPEC DECODE (sequential, stochastic) ===
        print("\n2. Spec Decode (sequential, stochastic acceptance)")
        let (specTps, specAcc, specRounds) = try await runSpec(
            draft: draftModel, target: targetModel, parallel: false)
        print(String(format: "   %.1f tok/s (accept=%.0f%%, rounds=%d)", specTps, specAcc, specRounds))

        // === 3. SSD (parallel, stochastic) ===
        print("\n3. SSD Parallel (stochastic acceptance)")
        let (ssdTps, ssdAcc, ssdRounds) = try await runSpec(
            draft: draftModel, target: targetModel, parallel: true)
        print(String(format: "   %.1f tok/s (accept=%.0f%%, rounds=%d)", ssdTps, ssdAcc, ssdRounds))

        // Summary
        print("\n" + String(repeating: "=", count: 60))
        print(String(format: "Vanilla: %.1f tok/s", vTps))
        print(String(format: "Spec:    %.1f tok/s (%.2fx)", specTps, specTps / vTps))
        print(String(format: "SSD:     %.1f tok/s (%.2fx)", ssdTps, ssdTps / vTps))
        print(String(format: "SSD vs Spec: %.2fx", ssdTps / specTps))
    }

    static func runSpec(draft: MLModel, target: MLModel, parallel: Bool) async throws -> (tps: Double, accRate: Double, rounds: Int) {
        let dState = draft.makeState()
        let tState = target.makeState()
        var tok: Int32 = 42
        var gen = 0
        var acc = 0
        var rounds = 0
        let start = CFAbsoluteTimeGetCurrent()

        while gen < maxTokens {
            rounds += 1
            let nd = min(draftCount, maxTokens - gen)

            // Draft: generate nd tokens with probabilities
            var draftToks: [Int32] = []
            var draftProbs: [Float] = []  // p_draft for each sampled token
            var dTok = tok
            for _ in 0..<nd {
                let r = try await draft.prediction(from: makeInput(token: dTok), using: dState)
                let logits = r.featureValue(for: "logits")!.multiArrayValue!
                let (sampled, prob) = sampleWithProb(logits, temp: temperature)
                draftToks.append(sampled)
                draftProbs.append(prob)
                dTok = sampled
            }

            if parallel {
                // Pre-speculate next round on draft (parallel with verify)
                async let _preSpec: Void = {
                    var pTok = dTok
                    for _ in 0..<nd {
                        let r = try await draft.prediction(from: makeInput(token: pTok), using: dState)
                        pTok = sampleTemp(r.featureValue(for: "logits")!.multiArrayValue!, temp: temperature)
                    }
                }()

                // Verify + stochastic accept
                var vTok = tok
                var accepted = 0
                for i in 0..<draftToks.count {
                    let r = try await target.prediction(from: makeInput(token: vTok), using: tState)
                    let targetLogits = r.featureValue(for: "logits")!.multiArrayValue!
                    let pTarget = getProb(targetLogits, token: draftToks[i], temp: temperature)
                    let pDraft = draftProbs[i]

                    // Stochastic acceptance
                    let acceptProb = min(1.0, pTarget / max(pDraft, 1e-10))
                    if Float.random(in: 0..<1) < acceptProb {
                        accepted += 1
                        vTok = draftToks[i]
                        gen += 1
                        acc += 1
                    } else {
                        // Reject: sample from residual distribution
                        vTok = sampleTemp(targetLogits, temp: temperature)
                        gen += 1
                        break
                    }
                    if gen >= maxTokens { break }
                }

                if accepted == draftToks.count && gen < maxTokens {
                    let r = try await target.prediction(from: makeInput(token: vTok), using: tState)
                    vTok = sampleTemp(r.featureValue(for: "logits")!.multiArrayValue!, temp: temperature)
                    gen += 1
                }

                try await _preSpec
                tok = vTok

            } else {
                // Sequential verify
                var vTok = tok
                var accepted = 0
                for i in 0..<draftToks.count {
                    let r = try await target.prediction(from: makeInput(token: vTok), using: tState)
                    let targetLogits = r.featureValue(for: "logits")!.multiArrayValue!
                    let pTarget = getProb(targetLogits, token: draftToks[i], temp: temperature)
                    let pDraft = draftProbs[i]

                    let acceptProb = min(1.0, pTarget / max(pDraft, 1e-10))
                    if Float.random(in: 0..<1) < acceptProb {
                        accepted += 1
                        vTok = draftToks[i]
                        gen += 1
                        acc += 1
                    } else {
                        vTok = sampleTemp(targetLogits, temp: temperature)
                        gen += 1
                        break
                    }
                    if gen >= maxTokens { break }
                }

                if accepted == draftToks.count && gen < maxTokens {
                    let r = try await target.prediction(from: makeInput(token: vTok), using: tState)
                    vTok = sampleTemp(r.featureValue(for: "logits")!.multiArrayValue!, temp: temperature)
                    gen += 1
                }

                tok = vTok
            }
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return (Double(gen) / elapsed, Double(acc) / Double(gen) * 100, rounds)
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

    static func sampleTemp(_ arr: MLMultiArray, temp: Float) -> Int32 {
        let (tok, _) = sampleWithProb(arr, temp: temp)
        return tok
    }

    static func sampleWithProb(_ arr: MLMultiArray, temp: Float) -> (Int32, Float) {
        let count = arr.count
        let ptr = arr.dataPointer.bindMemory(to: Float16.self, capacity: count)

        // Find max for numerical stability
        var maxVal: Float = -Float.infinity
        for i in 0..<count {
            let v = Float(ptr[i])
            if v > maxVal { maxVal = v }
        }

        // Softmax with temperature
        var probs = [Float](repeating: 0, count: count)
        var sum: Float = 0
        for i in 0..<count {
            let v = exp((Float(ptr[i]) - maxVal) / temp)
            probs[i] = v
            sum += v
        }
        for i in 0..<count { probs[i] /= sum }

        // Sample
        let r = Float.random(in: 0..<1)
        var cumSum: Float = 0
        for i in 0..<count {
            cumSum += probs[i]
            if cumSum > r {
                return (Int32(i), probs[i])
            }
        }
        return (Int32(count - 1), probs[count - 1])
    }

    static func getProb(_ arr: MLMultiArray, token: Int32, temp: Float) -> Float {
        let count = arr.count
        let ptr = arr.dataPointer.bindMemory(to: Float16.self, capacity: count)

        var maxVal: Float = -Float.infinity
        for i in 0..<count {
            let v = Float(ptr[i])
            if v > maxVal { maxVal = v }
        }

        // Just compute prob for the requested token
        var sum: Float = 0
        let targetVal = exp((Float(ptr[Int(token)]) - maxVal) / temp)
        for i in 0..<count {
            sum += exp((Float(ptr[i]) - maxVal) / temp)
        }
        return targetVal / sum
    }
}
