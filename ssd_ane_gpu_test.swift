// The critical test: does ANE overlap with GPU (Metal)?
// If yes: SSD works on Apple Silicon with CoreML draft + Metal target.
// If no: ANE and GPU share a bottleneck and can't run in parallel.

import Foundation
import CoreML

@main
struct ANEGPUTest {
    static func main() async throws {
        let modelPath = NSString(string: "~/models/qwen2.5-0.5b-coreml/Qwen2.5-0.5B-Instruct-4bit.mlmodelc").expandingTildeInPath
        let modelURL = URL(fileURLWithPath: modelPath)

        // ANE model (draft)
        let aneConfig = MLModelConfiguration()
        aneConfig.computeUnits = .cpuAndNeuralEngine  // Force ANE, no GPU
        let aneModel = try MLModel(contentsOf: modelURL, configuration: aneConfig)
        let aneState = aneModel.makeState()

        // GPU model (simulates target)
        let gpuConfig = MLModelConfiguration()
        gpuConfig.computeUnits = .cpuAndGPU  // Force GPU, no ANE
        let gpuModel = try MLModel(contentsOf: modelURL, configuration: gpuConfig)
        let gpuState = gpuModel.makeState()

        let input = try makeInput(token: 42)

        // Warmup
        print("Warming up...")
        for _ in 0..<10 {
            _ = try await aneModel.prediction(from: input, using: aneState)
            _ = try await gpuModel.prediction(from: input, using: gpuState)
        }

        let iterations = 200

        // Individual speeds
        let aneStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = try await aneModel.prediction(from: input, using: aneState)
        }
        let aneTime = (CFAbsoluteTimeGetCurrent() - aneStart) / Double(iterations) * 1000

        let gpuStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = try await gpuModel.prediction(from: input, using: gpuState)
        }
        let gpuTime = (CFAbsoluteTimeGetCurrent() - gpuStart) / Double(iterations) * 1000

        print(String(format: "ANE alone:  %.1f ms = %.0f tok/s", aneTime, 1000/aneTime))
        print(String(format: "GPU alone:  %.1f ms = %.0f tok/s", gpuTime, 1000/gpuTime))

        // Sequential
        let seqStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = try await aneModel.prediction(from: input, using: aneState)
            _ = try await gpuModel.prediction(from: input, using: gpuState)
        }
        let seqTime = (CFAbsoluteTimeGetCurrent() - seqStart) / Double(iterations) * 1000

        // Parallel: ANE + GPU simultaneously
        let parStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            async let aneResult = aneModel.prediction(from: input, using: aneState)
            let gpuResult = try await gpuModel.prediction(from: input, using: gpuState)
            let _ = try await aneResult
            let _ = gpuResult
        }
        let parTime = (CFAbsoluteTimeGetCurrent() - parStart) / Double(iterations) * 1000

        print(String(format: "\nSequential: %.1f ms", seqTime))
        print(String(format: "Parallel:   %.1f ms", parTime))
        print(String(format: "Speedup: %.2fx", seqTime / parTime))

        let theoreticalMax = max(aneTime, gpuTime)
        let saved = seqTime - parTime
        let maxSavable = seqTime - theoreticalMax
        let efficiency = maxSavable > 0 ? saved / maxSavable * 100 : 0
        print(String(format: "Overlap efficiency: %.0f%%", min(efficiency, 100)))

        if parTime < seqTime * 0.85 {
            print("\n*** ANE + GPU TRUE PARALLEL: SSD IS VIABLE ***")
        } else if parTime < seqTime * 0.95 {
            print("\nPartial overlap")
        } else {
            print("\nNo overlap — ANE and GPU may share a bottleneck")
        }
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
