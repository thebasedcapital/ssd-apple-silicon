// E2E Disaggregated: CoreML prefill → KV handoff → CoreML decode
// Saves KV cache to file, then a separate MLX process reads it.
// For now, measures prefill + KV export time.

import Foundation
import CoreML

@main
struct E2EDisagg {
    static func main() async throws {
        let prefillPath = NSString(string: "~/models/qwen2.5-0.5b-prefill/prefill.mlmodelc").expandingTildeInPath

        print("Loading prefill model...")
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let prefillModel = try MLModel(contentsOf: URL(fileURLWithPath: prefillPath), configuration: config)

        // Prefill 8 tokens
        let seqLen = 8
        let ids = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
        for i in 0..<seqLen { ids[i] = NSNumber(value: 42 + i) }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids)
        ])

        // Warmup
        for _ in 0..<3 { _ = try await prefillModel.prediction(from: input) }

        // Measure prefill + KV extraction
        let start = CFAbsoluteTimeGetCurrent()
        let result = try await prefillModel.prediction(from: input)
        let allK = result.featureValue(for: "all_k")!.multiArrayValue!
        let allV = result.featureValue(for: "all_v")!.multiArrayValue!

        // Write KV to /tmp as raw binary (for MLX to read)
        let kPtr = allK.dataPointer.assumingMemoryBound(to: UInt8.self)
        let vPtr = allV.dataPointer.assumingMemoryBound(to: UInt8.self)
        let kSize = allK.count * 2  // float16 = 2 bytes
        let vSize = allV.count * 2

        let kData = Data(bytes: kPtr, count: kSize)
        let vData = Data(bytes: vPtr, count: vSize)
        try kData.write(to: URL(fileURLWithPath: "/tmp/ssd_kv_k.bin"))
        try vData.write(to: URL(fileURLWithPath: "/tmp/ssd_kv_v.bin"))

        let prefillTime = (CFAbsoluteTimeGetCurrent() - start) * 1000

        // Write metadata
        let meta: [String: Any] = [
            "k_shape": allK.shape.map { $0.intValue },
            "v_shape": allV.shape.map { $0.intValue },
            "seq_len": seqLen,
            "prefill_ms": prefillTime,
        ]
        let metaData = try JSONSerialization.data(withJSONObject: meta)
        try metaData.write(to: URL(fileURLWithPath: "/tmp/ssd_kv_meta.json"))

        print(String(format: "Prefill + KV export: %.1f ms", prefillTime))
        print("K shape: \(allK.shape), V shape: \(allV.shape)")
        print("Saved to /tmp/ssd_kv_{k,v}.bin + meta.json")
        print(String(format: "K size: %.1f KB, V size: %.1f KB", Double(kSize)/1024, Double(vSize)/1024))
    }
}
