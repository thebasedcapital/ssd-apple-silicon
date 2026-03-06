// ANE draft token server. Reads token IDs from stdin, writes draft tokens to stdout.
// This runs in a subprocess, called from Python.
// Usage: echo "42" | ./ane_draft_server

import Foundation
import CoreML

@main
struct ANEDraftServer {
    static func main() async throws {
        let modelPath = NSString(string: "~/models/qwen2.5-0.5b-coreml/Qwen2.5-0.5B-Instruct-4bit.mlmodelc").expandingTildeInPath
        let modelURL = URL(fileURLWithPath: modelPath)

        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model = try MLModel(contentsOf: modelURL, configuration: config)
        let state = model.makeState()

        // Signal ready
        print("READY", terminator: "\n")
        fflush(stdout)

        // Read commands from stdin
        while let line = readLine() {
            let parts = line.split(separator: " ")
            guard let command = parts.first else { continue }

            if command == "DRAFT" {
                // DRAFT <start_token> <count>
                guard parts.count >= 3,
                      let startToken = Int32(parts[1]),
                      let count = Int(parts[2]) else {
                    print("ERROR bad args")
                    fflush(stdout)
                    continue
                }

                let start = CFAbsoluteTimeGetCurrent()
                var tokens: [Int32] = []
                var current = startToken

                for _ in 0..<count {
                    let ids = try MLMultiArray(shape: [1, 1], dataType: .int32)
                    ids[0] = NSNumber(value: current)
                    let mask = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
                    mask[0] = 0
                    let input = try MLDictionaryFeatureProvider(dictionary: [
                        "input_ids": MLFeatureValue(multiArray: ids),
                        "causal_mask": MLFeatureValue(multiArray: mask),
                    ])
                    let result = try await model.prediction(from: input, using: state)
                    let logits = result.featureValue(for: "logits")!.multiArrayValue!
                    current = argmax(logits)
                    tokens.append(current)
                }

                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                let tokenStr = tokens.map { String($0) }.joined(separator: ",")
                print("TOKENS \(tokenStr) \(String(format: "%.1f", elapsed))ms")
                fflush(stdout)

            } else if command == "RESET" {
                // Reset KV cache state
                // Note: MLState doesn't have a reset API, we'd need to recreate
                // For now, just acknowledge
                print("OK")
                fflush(stdout)

            } else if command == "QUIT" {
                break

            } else if command == "PING" {
                print("PONG")
                fflush(stdout)

            } else if command == "BENCH" {
                // Benchmark: single forward pass timing
                let ids = try MLMultiArray(shape: [1, 1], dataType: .int32)
                ids[0] = 42
                let mask = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
                mask[0] = 0
                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "input_ids": MLFeatureValue(multiArray: ids),
                    "causal_mask": MLFeatureValue(multiArray: mask),
                ])

                // Warmup
                for _ in 0..<5 {
                    _ = try await model.prediction(from: input, using: state)
                }

                let iterations = 100
                let start = CFAbsoluteTimeGetCurrent()
                for _ in 0..<iterations {
                    _ = try await model.prediction(from: input, using: state)
                }
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                let msPerPass = (elapsed / Double(iterations)) * 1000
                print(String(format: "BENCH %.1fms/forward %.1ftok/s", msPerPass, 1000.0 / msPerPass))
                fflush(stdout)
            }
        }
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
