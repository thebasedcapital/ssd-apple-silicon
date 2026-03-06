// Benchmark: CoreML ANE prefill speed for the stateless KV-output model.
// Measures how fast ANE can process N input tokens and output KV cache.

import Foundation
import CoreML

@main
struct BenchDisagg {
    static func main() async throws {
        let modelPath = NSString(string: "~/models/qwen2.5-0.5b-prefill/prefill.mlmodelc").expandingTildeInPath

        print("Loading CoreML prefill model...")
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model = try MLModel(contentsOf: URL(fileURLWithPath: modelPath), configuration: config)
        print("Loaded")

        // Print interface
        let desc = model.modelDescription
        print("Inputs:")
        for (n, f) in desc.inputDescriptionsByName {
            if let mc = f.multiArrayConstraint { print("  \(n): \(mc.shape) \(mc.dataType.rawValue)") }
        }
        print("Outputs:")
        for (n, f) in desc.outputDescriptionsByName {
            if let mc = f.multiArrayConstraint { print("  \(n): \(mc.shape) \(mc.dataType.rawValue)") }
            else { print("  \(n): \(f.type)") }
        }

        // Create input: 8 tokens
        let seqLen = 8
        let ids = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
        for i in 0..<seqLen { ids[i] = NSNumber(value: 42 + i) }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids)
        ])

        // Warmup
        print("\nWarming up...")
        for _ in 0..<3 {
            _ = try await model.prediction(from: input)
        }

        // Benchmark
        let iterations = 50
        print("Benchmarking \(iterations) prefill passes (seq_len=\(seqLen))...")
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            let result = try await model.prediction(from: input)
            // Force evaluation of outputs
            let _ = result.featureValue(for: "logits")!.multiArrayValue!
            let _ = result.featureValue(for: "all_k")!.multiArrayValue!
            let _ = result.featureValue(for: "all_v")!.multiArrayValue!
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let msPerPrefill = (elapsed / Double(iterations)) * 1000
        let toksPerSec = Double(seqLen) / (msPerPrefill / 1000.0)

        print(String(format: "  %.1f ms/prefill = %.0f tok/s (for %d tokens)", msPerPrefill, toksPerSec, seqLen))

        // Extract KV shape info
        let result = try await model.prediction(from: input)
        let allK = result.featureValue(for: "all_k")!.multiArrayValue!
        let allV = result.featureValue(for: "all_v")!.multiArrayValue!
        print(String(format: "\nKV output: all_k=%@ all_v=%@", allK.shape, allV.shape))

        // CPU comparison
        print("\nCPU comparison...")
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly
        let cpuModel = try MLModel(contentsOf: URL(fileURLWithPath: modelPath), configuration: cpuConfig)
        for _ in 0..<3 { _ = try await cpuModel.prediction(from: input) }
        let cpuStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations { _ = try await cpuModel.prediction(from: input) }
        let cpuElapsed = CFAbsoluteTimeGetCurrent() - cpuStart
        let cpuMs = (cpuElapsed / Double(iterations)) * 1000
        print(String(format: "  CPU: %.1f ms/prefill = %.0f tok/s", cpuMs, Double(seqLen) / (cpuMs / 1000.0)))
        print(String(format: "  ANE speedup: %.2fx", cpuMs / msPerPrefill))
    }
}
