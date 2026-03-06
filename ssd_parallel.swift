// SSD: ANE draft + GPU verify in parallel on Apple Silicon.
// Compile: swiftc -O -parse-as-library -framework CoreML -framework Foundation -framework Metal ssd_parallel.swift -o ssd_parallel

import Foundation
import CoreML

// Shared state between ANE draft thread and main verify loop
final class SharedState: @unchecked Sendable {
    let lock = NSLock()
    var predictionReady = false
    var predictedStartToken: Int32 = -1
    var draftTokens: [Int32] = []
}

@main
struct SSDParallel {
    static func main() async throws {
        let modelPath = NSString(string: "~/models/qwen2.5-0.5b-coreml/Qwen2.5-0.5B-Instruct-4bit.mlmodelc").expandingTildeInPath
        let modelURL = URL(fileURLWithPath: modelPath)

        // Load TWO copies: one for ANE (speculation), one for GPU verify baseline
        print("Loading CoreML models...")
        let aneConfig = MLModelConfiguration()
        aneConfig.computeUnits = .all  // ANE preferred
        let aneModel = try MLModel(contentsOf: modelURL, configuration: aneConfig)

        let gpuConfig = MLModelConfiguration()
        gpuConfig.computeUnits = .cpuAndGPU  // GPU for "target" simulation
        let gpuModel = try MLModel(contentsOf: modelURL, configuration: gpuConfig)

        print("Both models loaded\n")

        // === BENCHMARK 1: Sequential (standard spec decode simulation) ===
        // Draft on GPU -> Verify on GPU -> repeat
        let seqTokens = 128
        let draftCount = 2

        print("=== Sequential (draft+verify on same compute) ===")
        let gpuState = gpuModel.makeState()
        var seqGenerated = 0
        let seqStart = CFAbsoluteTimeGetCurrent()

        var currentToken: Int32 = 42 // Start token
        while seqGenerated < seqTokens {
            // Draft phase: generate draftCount tokens
            var draftToks: [Int32] = []
            for _ in 0..<draftCount {
                let input = try makeInput(token: currentToken)
                let result = try await gpuModel.prediction(from: input, using: gpuState)
                let logits = result.featureValue(for: "logits")!.multiArrayValue!
                let nextTok = argmax(logits)
                draftToks.append(nextTok)
                currentToken = nextTok
            }

            // Verify phase: one more forward pass (simulates target check)
            let verifyInput = try makeInput(token: currentToken)
            let _ = try await gpuModel.prediction(from: verifyInput, using: gpuState)

            seqGenerated += draftCount
        }

        let seqElapsed = CFAbsoluteTimeGetCurrent() - seqStart
        let seqTps = Double(seqGenerated) / seqElapsed
        print(String(format: "  %d tokens in %.2fs = %.1f tok/s\n", seqGenerated, seqElapsed, seqTps))

        // === BENCHMARK 2: Parallel SSD (ANE draft overlaps with GPU verify) ===
        print("=== Parallel SSD (ANE draft || GPU verify) ===")
        let aneState = aneModel.makeState()
        let gpuState2 = gpuModel.makeState()
        let shared = SharedState()

        var parGenerated = 0
        var parCurrentToken: Int32 = 42
        let parStart = CFAbsoluteTimeGetCurrent()

        // First round: draft normally (no speculation yet)
        var prevDraftToks: [Int32] = []
        for _ in 0..<draftCount {
            let input = try makeInput(token: parCurrentToken)
            let result = try await aneModel.prediction(from: input, using: aneState)
            let logits = result.featureValue(for: "logits")!.multiArrayValue!
            let nextTok = argmax(logits)
            prevDraftToks.append(nextTok)
            parCurrentToken = nextTok
        }

        while parGenerated < seqTokens {
            // === PARALLEL PHASE ===
            // GPU: verify current draft tokens
            // ANE: simultaneously pre-draft for next round

            let verifyToken = parCurrentToken
            let speculateFrom = parCurrentToken // Predict: next round starts from last draft token

            // Start ANE speculation in background
            shared.lock.lock()
            shared.predictionReady = false
            shared.lock.unlock()

            let aneTask = Task.detached(priority: .high) {
                var specToks: [Int32] = []
                var tok = speculateFrom
                for _ in 0..<draftCount {
                    let input = try makeInput(token: tok)
                    let result = try await aneModel.prediction(from: input, using: aneState)
                    let logits = result.featureValue(for: "logits")!.multiArrayValue!
                    tok = argmax(logits)
                    specToks.append(tok)
                }
                shared.lock.lock()
                shared.draftTokens = specToks
                shared.predictedStartToken = speculateFrom
                shared.predictionReady = true
                shared.lock.unlock()
            }

            // GPU: verify (runs simultaneously with ANE task above)
            let verifyInput = try makeInput(token: verifyToken)
            let _ = try await gpuModel.prediction(from: verifyInput, using: gpuState2)

            // Wait for ANE to finish (should already be done — it's faster)
            _ = await aneTask.result

            parGenerated += draftCount

            // Check if speculation is usable
            shared.lock.lock()
            let ready = shared.predictionReady
            let specToks = shared.draftTokens
            let specStart = shared.predictedStartToken
            shared.lock.unlock()

            if ready && specStart == parCurrentToken && !specToks.isEmpty {
                // HIT: use pre-speculated tokens, skip draft step
                prevDraftToks = specToks
                parCurrentToken = specToks.last!
            } else {
                // MISS: draft normally
                prevDraftToks = []
                for _ in 0..<draftCount {
                    let input = try makeInput(token: parCurrentToken)
                    let result = try await aneModel.prediction(from: input, using: aneState)
                    let logits = result.featureValue(for: "logits")!.multiArrayValue!
                    let nextTok = argmax(logits)
                    prevDraftToks.append(nextTok)
                    parCurrentToken = nextTok
                }
            }
        }

        let parElapsed = CFAbsoluteTimeGetCurrent() - parStart
        let parTps = Double(parGenerated) / parElapsed
        print(String(format: "  %d tokens in %.2fs = %.1f tok/s", parGenerated, parElapsed, parTps))
        print(String(format: "  Speedup: %.2fx over sequential\n", parTps / seqTps))
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
