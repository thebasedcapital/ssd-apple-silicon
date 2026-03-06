// Real overlap test: ANE draft + CPU "target" running in parallel.
// CPU simulates a slower target model. Measures if ANE work is truly hidden.
//
// If overlap works: parallel time ≈ max(ANE_time, CPU_time)
// If no overlap:    parallel time ≈ ANE_time + CPU_time

import Foundation
import CoreML

@main
struct SSDOverlapTest {
    static func main() async throws {
        let modelPath = NSString(string: "~/models/qwen2.5-0.5b-coreml/Qwen2.5-0.5B-Instruct-4bit.mlmodelc").expandingTildeInPath
        let modelURL = URL(fileURLWithPath: modelPath)

        // ANE model (draft) — runs on Neural Engine
        let aneConfig = MLModelConfiguration()
        aneConfig.computeUnits = .all
        let aneModel = try MLModel(contentsOf: modelURL, configuration: aneConfig)
        let aneState = aneModel.makeState()

        // CPU model (simulates slow target) — runs on CPU only
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly
        let cpuModel = try MLModel(contentsOf: modelURL, configuration: cpuConfig)
        let cpuState = cpuModel.makeState()

        let input = try makeInput(token: 42)

        // Warmup
        print("Warming up...")
        for _ in 0..<10 {
            _ = try await aneModel.prediction(from: input, using: aneState)
            _ = try await cpuModel.prediction(from: input, using: cpuState)
        }

        let iterations = 200

        // === Measure individual speeds ===
        let aneStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = try await aneModel.prediction(from: input, using: aneState)
        }
        let aneTime = (CFAbsoluteTimeGetCurrent() - aneStart) / Double(iterations) * 1000

        let cpuStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = try await cpuModel.prediction(from: input, using: cpuState)
        }
        let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) / Double(iterations) * 1000

        print(String(format: "ANE alone:  %.1f ms/forward = %.0f tok/s", aneTime, 1000/aneTime))
        print(String(format: "CPU alone:  %.1f ms/forward = %.0f tok/s", cpuTime, 1000/cpuTime))
        print(String(format: "Sequential: %.1f ms (sum)", aneTime + cpuTime))
        print(String(format: "Ideal parallel: %.1f ms (max)", max(aneTime, cpuTime)))

        // === Sequential: ANE then CPU ===
        let seqStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = try await aneModel.prediction(from: input, using: aneState)
            _ = try await cpuModel.prediction(from: input, using: cpuState)
        }
        let seqTime = (CFAbsoluteTimeGetCurrent() - seqStart) / Double(iterations) * 1000

        // === Parallel: ANE and CPU simultaneously ===
        let parStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            // Launch ANE in a detached task
            async let aneResult = aneModel.prediction(from: input, using: aneState)
            // CPU runs on current thread
            let cpuResult = try await cpuModel.prediction(from: input, using: cpuState)
            // Wait for ANE
            let _ = try await aneResult
            let _ = cpuResult
        }
        let parTime = (CFAbsoluteTimeGetCurrent() - parStart) / Double(iterations) * 1000

        print(String(format: "\nSequential (measured): %.1f ms", seqTime))
        print(String(format: "Parallel (measured):   %.1f ms", parTime))
        print(String(format: "Speedup: %.2fx", seqTime / parTime))

        let theoreticalMax = max(aneTime, cpuTime)
        let overlapEfficiency = (seqTime - parTime) / (seqTime - theoreticalMax) * 100
        print(String(format: "Overlap efficiency: %.0f%%", min(overlapEfficiency, 100)))

        if parTime < seqTime * 0.85 {
            print("\n*** TRUE PARALLEL EXECUTION: ANE + CPU confirmed ***")
        } else if parTime < seqTime * 0.95 {
            print("\nPartial overlap detected")
        } else {
            print("\nNo meaningful overlap — likely serialized")
        }

        // === Now test with draft=2 tokens (realistic SSD scenario) ===
        print("\n--- Realistic SSD Scenario (draft=2) ---")
        let draftCount = 2

        // Sequential: draft 2 tokens (ANE), then verify (CPU)
        let realSeqStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<(iterations / draftCount) {
            // Draft phase: 2 tokens on ANE
            for _ in 0..<draftCount {
                _ = try await aneModel.prediction(from: input, using: aneState)
            }
            // Verify phase: 1 forward pass on CPU (simulating target batch verify)
            _ = try await cpuModel.prediction(from: input, using: cpuState)
        }
        let realSeqTime = (CFAbsoluteTimeGetCurrent() - realSeqStart) / Double(iterations / draftCount) * 1000

        // SSD: while CPU verifies round T, ANE drafts for round T+1
        let realParStart = CFAbsoluteTimeGetCurrent()
        // First round: draft normally
        for _ in 0..<draftCount {
            _ = try await aneModel.prediction(from: input, using: aneState)
        }
        for round in 0..<(iterations / draftCount) {
            // Parallel: CPU verify + ANE draft next round
            async let aneDraft: Void = {
                for _ in 0..<draftCount {
                    _ = try await aneModel.prediction(from: input, using: aneState)
                }
            }()
            let _ = try await cpuModel.prediction(from: input, using: cpuState)
            try await aneDraft
        }
        let realParTime = (CFAbsoluteTimeGetCurrent() - realParStart) / Double(iterations / draftCount) * 1000

        print(String(format: "Sequential (draft+verify): %.1f ms/round", realSeqTime))
        print(String(format: "SSD parallel:              %.1f ms/round", realParTime))
        print(String(format: "SSD speedup: %.2fx", realSeqTime / realParTime))
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
}
