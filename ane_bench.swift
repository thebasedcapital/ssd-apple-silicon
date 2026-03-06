import Foundation
import CoreML

@main
struct ANEBench {
    static func main() async throws {
        let modelPath = NSString(string: "~/models/qwen2.5-0.5b-coreml-matched/draft.mlmodelc").expandingTildeInPath
        let modelURL = URL(fileURLWithPath: modelPath)

        print("Loading CoreML model...")
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model = try MLModel(contentsOf: modelURL, configuration: config)
        print("Loaded")

        // Print ALL inputs/outputs/states
        let desc = model.modelDescription
        print("\nInputs:")
        for (name, feat) in desc.inputDescriptionsByName.sorted(by: { $0.key < $1.key }) {
            if let mc = feat.multiArrayConstraint {
                print("  \(name): shape=\(mc.shape) type=\(mc.dataType.rawValue)")
            } else if feat.isOptional {
                print("  \(name): optional \(feat.type)")
            } else {
                print("  \(name): \(feat.type)")
            }
        }
        print("\nOutputs:")
        for (name, feat) in desc.outputDescriptionsByName.sorted(by: { $0.key < $1.key }) {
            print("  \(name): \(feat.type)")
        }
        print("\nStates:")
        for (name, feat) in desc.stateDescriptionsByName.sorted(by: { $0.key < $1.key }) {
            print("  \(name): \(feat)")
        }

        // Create state for stateful model
        let state = try model.makeState()

        // Build non-state inputs only
        var inputs: [String: MLFeatureValue] = [:]
        for (name, feat) in desc.inputDescriptionsByName {
            // Skip state inputs (key_cache, value_cache)
            if name.contains("cache") { continue }

            guard let mc = feat.multiArrayConstraint else { continue }
            let arr = try MLMultiArray(shape: mc.shape, dataType: mc.dataType)

            if name == "input_ids" {
                arr[0] = 42
            } else if name == "causal_mask" {
                // Fill with 0 (no masking for single token)
                for i in 0..<arr.count { arr[i] = 0 }
            }

            inputs[name] = MLFeatureValue(multiArray: arr)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)

        // Warmup
        print("\nWarming up...")
        for _ in 0..<3 {
            _ = try await model.prediction(from: provider, using: state)
        }

        // Benchmark
        let iterations = 100
        print("Benchmarking \(iterations) forward passes...")
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = try await model.prediction(from: provider, using: state)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let msPerPass = (elapsed / Double(iterations)) * 1000
        print(String(format: "  ANE/GPU: %.1f ms/forward = %.1f tok/s", msPerPass, 1000.0 / msPerPass))

        // CPU-only comparison
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly
        let cpuModel = try MLModel(contentsOf: modelURL, configuration: cpuConfig)
        let cpuState = try cpuModel.makeState()
        for _ in 0..<3 { _ = try await cpuModel.prediction(from: provider, using: cpuState) }
        let cpuStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations { _ = try await cpuModel.prediction(from: provider, using: cpuState) }
        let cpuElapsed = CFAbsoluteTimeGetCurrent() - cpuStart
        let cpuMs = (cpuElapsed / Double(iterations)) * 1000
        print(String(format: "  CPU:     %.1f ms/forward = %.1f tok/s", cpuMs, 1000.0 / cpuMs))
        print(String(format: "  Speedup: %.2fx", cpuMs / msPerPass))
    }
}
