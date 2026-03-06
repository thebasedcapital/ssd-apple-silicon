// Benchmark: CoreML 3B ANE prefill speed + KV export
// Then compare with pure MLX prefill timing

import Foundation
import CoreML

@main
struct Bench3BPrefill {
    static func main() async throws {
        let modelPath = "/Volumes/QWER/models/qwen2.5-3b-prefill/prefill.mlmodelc"

        print("Loading 3B CoreML prefill model...")
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model = try MLModel(contentsOf: URL(fileURLWithPath: modelPath), configuration: config)
        print("Loaded")

        // Print interface
        for (n, f) in model.modelDescription.outputDescriptionsByName {
            if let mc = f.multiArrayConstraint { print("  \(n): \(mc.shape)") }
        }

        // Create input: 40 tokens
        let seqLen = 40
        let ids = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
        for i in 0..<seqLen { ids[i] = NSNumber(value: 42 + i) }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids)
        ])

        // Warmup
        print("Warming up...")
        for _ in 0..<2 {
            _ = try await model.prediction(from: input)
        }

        // Benchmark prefill
        let iterations = 10
        print("Benchmarking \(iterations) prefill passes (seq_len=\(seqLen))...")
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            let result = try await model.prediction(from: input)
            let _ = result.featureValue(for: "logits")!.multiArrayValue!
            let _ = result.featureValue(for: "all_k")!.multiArrayValue!
            let _ = result.featureValue(for: "all_v")!.multiArrayValue!
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let msPerPrefill = (elapsed / Double(iterations)) * 1000
        let toksPerSec = Double(seqLen) / (msPerPrefill / 1000.0)
        print(String(format: "  CoreML 3B prefill: %.1f ms = %.0f tok/s (%d tokens)", msPerPrefill, toksPerSec, seqLen))

        // Export KV for MLX
        let result = try await model.prediction(from: input)
        let allK = result.featureValue(for: "all_k")!.multiArrayValue!
        let allV = result.featureValue(for: "all_v")!.multiArrayValue!

        let kSize = allK.count * 2
        let vSize = allV.count * 2
        let kData = Data(bytes: allK.dataPointer, count: kSize)
        let vData = Data(bytes: allV.dataPointer, count: vSize)
        try kData.write(to: URL(fileURLWithPath: "/tmp/ssd_3b_k.bin"))
        try vData.write(to: URL(fileURLWithPath: "/tmp/ssd_3b_v.bin"))

        let meta: [String: Any] = [
            "k_shape": allK.shape.map { $0.intValue },
            "v_shape": allV.shape.map { $0.intValue },
            "seq_len": seqLen,
            "prefill_ms": msPerPrefill,
        ]
        try JSONSerialization.data(withJSONObject: meta).write(to: URL(fileURLWithPath: "/tmp/ssd_3b_meta.json"))

        print(String(format: "KV exported: k=%@ v=%@ (%.0f KB each)", allK.shape, allV.shape, Double(kSize)/1024))
    }
}
