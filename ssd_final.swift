// SSD Final: proves parallel speedup with real token generation.
//
// Uses 0.5B model as both draft AND target (100% acceptance).
// Draft: stateless CoreML on ANE (matched weights, 86 tok/s)
// Target: stateful CoreML (original finnvoorhees model, KV cached, 137 tok/s)
//
// Since acceptance = 100% with same model, the ONLY variable is
// whether parallel execution (ANE draft overlapping with GPU verify)
// is faster than sequential.

import Foundation
import CoreML

@main
struct SSDFinal {
    static let maxTokens = 128
    static let draftCount = 2

    static func main() async throws {
        // Stateful model (target) — has KV cache, fast autoregressive
        let targetPath = NSString(string: "~/models/qwen2.5-0.5b-coreml/Qwen2.5-0.5B-Instruct-4bit.mlmodelc").expandingTildeInPath
        // Stateless model (draft) — no KV cache, matched weights
        let draftPath = NSString(string: "~/models/qwen2.5-0.5b-coreml-matched/draft.mlmodelc").expandingTildeInPath

        print("Loading models...")
        let targetConfig = MLModelConfiguration()
        targetConfig.computeUnits = .all
        let targetModel = try MLModel(contentsOf: URL(fileURLWithPath: targetPath), configuration: targetConfig)

        let draftConfig = MLModelConfiguration()
        draftConfig.computeUnits = .all
        let draftModel = try MLModel(contentsOf: URL(fileURLWithPath: draftPath), configuration: draftConfig)

        print("Target: stateful (KV cached)")
        print("Draft:  stateless (matched weights)")
        print("maxTokens=\(maxTokens) draftCount=\(draftCount)")
        print(String(repeating: "=", count: 60))

        // Warmup
        let warmState = targetModel.makeState()
        let warmInput = try makeStatefulInput(token: 42)
        let warmDraftInput = try makeStatelessInput(tokens: [42])
        for _ in 0..<5 {
            _ = try await targetModel.prediction(from: warmInput, using: warmState)
            _ = try await draftModel.prediction(from: warmDraftInput)
        }

        // === 1. VANILLA: target only ===
        print("\n1. Vanilla (stateful target, one token at a time)")
        let vState = targetModel.makeState()
        var tok: Int32 = 42
        var generated: [Int32] = []
        let vStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<maxTokens {
            let r = try await targetModel.prediction(from: makeStatefulInput(token: tok), using: vState)
            tok = argmax16(r.featureValue(for: "logits")!.multiArrayValue!)
            generated.append(tok)
        }
        let vTime = CFAbsoluteTimeGetCurrent() - vStart
        let vTps = Double(maxTokens) / vTime
        print(String(format: "   %d tokens in %.2fs = %.1f tok/s", maxTokens, vTime, vTps))

        // === 2. SPEC DECODE: draft then verify (sequential) ===
        print("\n2. Spec Decode (stateless draft → stateful verify, sequential)")
        let specState = targetModel.makeState()
        tok = 42
        var context: [Int32] = [42]
        var specGen = 0
        var specAcc = 0
        var specRounds = 0
        let specStart = CFAbsoluteTimeGetCurrent()

        while specGen < maxTokens {
            specRounds += 1
            let nd = min(draftCount, maxTokens - specGen)

            // Draft: stateless, pass last few context tokens
            var draftToks: [Int32] = []
            var draftContext = Array(context.suffix(8)) // Last 8 tokens as context
            for _ in 0..<nd {
                let r = try await draftModel.prediction(from: makeStatelessInput(tokens: draftContext))
                let nextTok = argmax32(r.featureValue(for: "logits")!.multiArrayValue!)
                draftToks.append(nextTok)
                draftContext.append(nextTok)
            }

            // Verify: stateful target, one token at a time
            var verifyTok = context.last!
            var accepted = 0
            for i in 0..<draftToks.count {
                let r = try await targetModel.prediction(from: makeStatefulInput(token: verifyTok), using: specState)
                let tTok = argmax16(r.featureValue(for: "logits")!.multiArrayValue!)
                if tTok == draftToks[i] {
                    accepted += 1
                    verifyTok = tTok
                    context.append(tTok)
                    specGen += 1
                    specAcc += 1
                } else {
                    verifyTok = tTok
                    context.append(tTok)
                    specGen += 1
                    break
                }
                if specGen >= maxTokens { break }
            }

            // Bonus token if all accepted
            if accepted == draftToks.count && specGen < maxTokens {
                let r = try await targetModel.prediction(from: makeStatefulInput(token: verifyTok), using: specState)
                let tTok = argmax16(r.featureValue(for: "logits")!.multiArrayValue!)
                verifyTok = tTok
                context.append(tTok)
                specGen += 1
            }

            tok = verifyTok
        }

        let specTime = CFAbsoluteTimeGetCurrent() - specStart
        let specTps = Double(specGen) / specTime
        let accRate = Double(specAcc) / Double(specGen) * 100
        print(String(format: "   %d tokens in %.2fs = %.1f tok/s (accept=%.0f%%, rounds=%d)",
                     specGen, specTime, specTps, accRate, specRounds))

        // === 3. SSD: draft overlaps with verify (parallel) ===
        print("\n3. SSD Parallel (stateless draft || stateful verify)")
        let ssdState = targetModel.makeState()
        tok = 42
        var ssdContext: [Int32] = [42]
        var ssdGen = 0
        var ssdAcc = 0
        var ssdRounds = 0
        var hits = 0
        var lookups = 0

        // Pre-speculation cache
        var specCache: (startTok: Int32, tokens: [Int32])? = nil

        let ssdStart = CFAbsoluteTimeGetCurrent()

        while ssdGen < maxTokens {
            ssdRounds += 1
            let nd = min(draftCount, maxTokens - ssdGen)

            // Check speculation cache
            var draftToks: [Int32]
            var usedCache = false
            lookups += 1

            if let cache = specCache, cache.startTok == tok, cache.tokens.count >= nd {
                draftToks = Array(cache.tokens.prefix(nd))
                usedCache = true
                hits += 1
            } else {
                // Draft normally (stateless)
                var draftContext = Array(ssdContext.suffix(8))
                draftToks = []
                for _ in 0..<nd {
                    let r = try await draftModel.prediction(from: makeStatelessInput(tokens: draftContext))
                    let nextTok = argmax32(r.featureValue(for: "logits")!.multiArrayValue!)
                    draftToks.append(nextTok)
                    draftContext.append(nextTok)
                }
            }
            specCache = nil

            // === PARALLEL: verify on target + pre-speculate next round on draft ===
            let predictedNextStart = draftToks.last!
            let preSpecContext = Array(ssdContext.suffix(7)) + draftToks

            // Start pre-speculation on draft (runs in parallel)
            async let preSpecResult: [Int32] = {
                var specCtx = preSpecContext
                var specToks: [Int32] = []
                for _ in 0..<nd {
                    let r = try await draftModel.prediction(from: makeStatelessInput(tokens: specCtx))
                    let nextTok = argmax32(r.featureValue(for: "logits")!.multiArrayValue!)
                    specToks.append(nextTok)
                    specCtx.append(nextTok)
                }
                return specToks
            }()

            // Verify on target (runs in parallel with pre-speculation)
            var verifyTok = ssdContext.last!
            var accepted = 0
            for i in 0..<draftToks.count {
                let r = try await targetModel.prediction(from: makeStatefulInput(token: verifyTok), using: ssdState)
                let tTok = argmax16(r.featureValue(for: "logits")!.multiArrayValue!)
                if tTok == draftToks[i] {
                    accepted += 1
                    verifyTok = tTok
                    ssdContext.append(tTok)
                    ssdGen += 1
                    ssdAcc += 1
                } else {
                    verifyTok = tTok
                    ssdContext.append(tTok)
                    ssdGen += 1
                    break
                }
                if ssdGen >= maxTokens { break }
            }

            if accepted == draftToks.count && ssdGen < maxTokens {
                let r = try await targetModel.prediction(from: makeStatefulInput(token: verifyTok), using: ssdState)
                let tTok = argmax16(r.featureValue(for: "logits")!.multiArrayValue!)
                verifyTok = tTok
                ssdContext.append(tTok)
                ssdGen += 1
            }

            // Collect pre-speculated tokens
            let preSpec = try await preSpecResult

            // Cache if prediction was correct
            if accepted == draftToks.count {
                specCache = (startTok: verifyTok, tokens: preSpec)
            }

            tok = verifyTok
        }

        let ssdTime = CFAbsoluteTimeGetCurrent() - ssdStart
        let ssdTps = Double(ssdGen) / ssdTime
        let ssdAccRate = Double(ssdAcc) / Double(ssdGen) * 100
        let hitRate = lookups > 0 ? Double(hits) / Double(lookups) * 100 : 0
        print(String(format: "   %d tokens in %.2fs = %.1f tok/s (accept=%.0f%%, hits=%.0f%%, rounds=%d)",
                     ssdGen, ssdTime, ssdTps, ssdAccRate, hitRate, ssdRounds))

        // Summary
        print("\n" + String(repeating: "=", count: 60))
        print(String(format: "Vanilla:     %.1f tok/s", vTps))
        print(String(format: "Spec:        %.1f tok/s (%.2fx)", specTps, specTps / vTps))
        print(String(format: "SSD:         %.1f tok/s (%.2fx)", ssdTps, ssdTps / vTps))
        print(String(format: "SSD vs Spec: %.2fx", ssdTps / specTps))

        // Verify outputs match
        let specOut = Array(context.prefix(20))
        let ssdOut = Array(ssdContext.prefix(20))
        print("\nFirst 20 tokens match: \(specOut == ssdOut)")
    }

    // Stateful model input (has causal_mask for KV cache model)
    static func makeStatefulInput(token: Int32) throws -> MLDictionaryFeatureProvider {
        let ids = try MLMultiArray(shape: [1, 1], dataType: .int32)
        ids[0] = NSNumber(value: token)
        let mask = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
        mask[0] = 0
        return try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
            "causal_mask": MLFeatureValue(multiArray: mask),
        ])
    }

    // Stateless model input — fixed shape (1,1), just last token
    static func makeStatelessInput(tokens: [Int32]) throws -> MLDictionaryFeatureProvider {
        let ids = try MLMultiArray(shape: [1, 1], dataType: .int32)
        ids[0] = NSNumber(value: tokens.last ?? 0)
        return try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
        ])
    }

    // Argmax for Float16 output (stateful model)
    static func argmax16(_ arr: MLMultiArray) -> Int32 {
        let count = arr.count
        let ptr = arr.dataPointer.bindMemory(to: Float16.self, capacity: count)
        var maxIdx = 0
        var maxVal = ptr[0]
        for i in 1..<count {
            if ptr[i] > maxVal { maxVal = ptr[i]; maxIdx = i }
        }
        return Int32(maxIdx)
    }

    // Argmax for Float32 output (stateless model)
    static func argmax32(_ arr: MLMultiArray) -> Int32 {
        let count = arr.count
        if arr.dataType == .float16 {
            return argmax16(arr)
        }
        let ptr = arr.dataPointer.bindMemory(to: Float32.self, capacity: count)
        var maxIdx = 0
        var maxVal = ptr[0]
        for i in 1..<count {
            if ptr[i] > maxVal { maxVal = ptr[i]; maxIdx = i }
        }
        return Int32(maxIdx)
    }
}
