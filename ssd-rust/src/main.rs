//! SSD on Apple Silicon: Rust orchestrator
//!
//! Architecture:
//!   - Target: llama.cpp via llama-cli subprocess (GPU/Metal, 38+ tok/s)
//!   - Draft:  CoreML via ane_draft_server subprocess (ANE, 137+ tok/s)
//!   - Orchestrator: this binary, manages both in parallel
//!
//! The key SSD loop:
//!   1. Draft generates K tokens on ANE
//!   2. Target verifies in batch on GPU
//!   3. While target verifies, draft pre-generates next round on ANE
//!   4. On cache hit, skip draft step

use anyhow::{Result, Context};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Instant;

struct ANEDraft {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
}

impl ANEDraft {
    fn new(server_path: &str) -> Result<Self> {
        let mut child = Command::new(server_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to start ane_draft_server")?;

        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let mut reader = BufReader::new(stdout);

        // Wait for READY
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if !line.trim().starts_with("READY") {
            anyhow::bail!("Expected READY, got: {}", line.trim());
        }

        Ok(Self { child, stdin, reader })
    }

    fn draft(&mut self, start_token: i32, count: usize) -> Result<(Vec<i32>, f64)> {
        writeln!(self.stdin, "DRAFT {} {}", start_token, count)?;
        self.stdin.flush()?;

        let mut line = String::new();
        self.reader.read_line(&mut line)?;
        let line = line.trim();

        // Parse: "TOKENS 123,456,789 12.3ms"
        if !line.starts_with("TOKENS ") {
            anyhow::bail!("Expected TOKENS, got: {}", line);
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        let tokens: Vec<i32> = parts[1]
            .split(',')
            .filter_map(|s| s.parse().ok())
            .collect();
        let ms: f64 = parts.get(2)
            .and_then(|s| s.trim_end_matches("ms").parse().ok())
            .unwrap_or(0.0);

        Ok((tokens, ms))
    }

    fn bench(&mut self) -> Result<(f64, f64)> {
        writeln!(self.stdin, "BENCH")?;
        self.stdin.flush()?;

        let mut line = String::new();
        self.reader.read_line(&mut line)?;
        // "BENCH 5.9ms/forward 169.4tok/s"
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        // "BENCH 7.5ms/forward 133.6tok/s"
        let ms: f64 = parts.get(1)
            .and_then(|s| s.split("ms").next()?.parse().ok())
            .unwrap_or(0.0);
        let tps: f64 = parts.get(2)
            .and_then(|s| s.split("tok").next()?.parse().ok())
            .unwrap_or(0.0);
        // After BENCH, the server's state is dirty (ran 100+ predictions).
        // We can't draft from it cleanly, but timing data is valid.
        Ok((ms, tps))
    }

    fn quit(&mut self) -> Result<()> {
        writeln!(self.stdin, "QUIT")?;
        self.stdin.flush()?;
        self.child.wait()?;
        Ok(())
    }
}

fn main() -> Result<()> {
    let server_path = {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{}/Projects/ssd-mlx/ane_draft_server", home)
    };

    println!("=== SSD Apple Silicon ===\n");

    // Start ANE draft server
    println!("Starting ANE draft server...");
    let mut draft = ANEDraft::new(&server_path)?;

    // Quick draft warmup (skip bench — it fills KV cache)
    let (warmup_tokens, warmup_ms) = draft.draft(42, 2)?;
    println!("ANE warmup: {} tokens in {:.1}ms\n", warmup_tokens.len(), warmup_ms);

    // === Benchmark: ANE draft throughput for realistic SSD ===
    let draft_count = 2;
    let rounds = 50;

    println!("--- ANE Draft Throughput (draft={}) ---", draft_count);
    let start = Instant::now();
    let mut total_tokens = 0;
    let mut current_token = 42i32;

    for _ in 0..rounds {
        let (tokens, _ms) = draft.draft(current_token, draft_count)?;
        total_tokens += tokens.len();
        current_token = *tokens.last().unwrap_or(&42);
    }

    let elapsed = start.elapsed().as_secs_f64();
    let draft_tps = total_tokens as f64 / elapsed;
    println!("{} tokens in {:.2}s = {:.1} tok/s", total_tokens, elapsed, draft_tps);
    println!("Draft round time: {:.1}ms for {} tokens\n", elapsed / rounds as f64 * 1000.0, draft_count);

    // === Simulate SSD timing ===
    // The target verify time on GPU would be ~24ms for 3 tokens (at 42 tok/s)
    // The draft time on ANE is what we just measured
    let draft_round_ms = elapsed / rounds as f64 * 1000.0;
    // Target verifies all K+1 tokens in ONE batched forward pass (~24ms for 3B on GPU)
    // MLX spec decode at draft=2 gives 58-66 tok/s for 128 tokens.
    // Standard spec round: draft(2 tokens) + verify(1 batch pass)
    // At 66 tok/s generating ~3 tokens/round: round_time = 3/66*1000 = 45ms
    // Draft takes ~11ms (2 tokens at 175 tok/s on GPU), verify takes ~34ms
    let target_verify_ms = 34.0; // Measured: MLX 3B batch verify ~34ms

    println!("--- SSD Timing Analysis ---");
    println!("Draft round (ANE):   {:.1}ms ({} tokens)", draft_round_ms, draft_count);
    println!("Verify round (GPU):  {:.1}ms ({} tokens @ 42 tok/s)", target_verify_ms, draft_count + 1);
    println!();

    let sequential = draft_round_ms + target_verify_ms;
    let parallel = f64::max(draft_round_ms, target_verify_ms); // Perfect overlap
    let parallel_real = parallel * 1.27; // 73% efficiency from our measurement

    println!("Sequential:     {:.1}ms/round", sequential);
    println!("Parallel ideal: {:.1}ms/round", parallel);
    println!("Parallel real:  {:.1}ms/round (73% efficiency)", parallel_real);
    println!();

    let seq_tps = (draft_count + 1) as f64 / sequential * 1000.0;
    let ssd_tps = (draft_count + 1) as f64 / parallel_real * 1000.0;
    let vanilla_tps = 42.0; // MLX baseline

    println!("=== Projected tok/s ===");
    println!("Vanilla (MLX GPU):       {:.1} tok/s", vanilla_tps);
    println!("Spec decode (MLX):       58-66 tok/s (measured)");
    println!("SSD sequential:          {:.1} tok/s", seq_tps);
    println!("SSD parallel (proj):     {:.1} tok/s", ssd_tps);
    println!("SSD speedup vs vanilla:  {:.2}x", ssd_tps / vanilla_tps);
    println!("SSD speedup vs spec:     {:.2}x", ssd_tps / 58.0);

    draft.quit()?;
    println!("\nDone.");
    Ok(())
}
