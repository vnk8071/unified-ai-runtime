// SPDX-License-Identifier: Apache-2.0
// Runs a model through the QNN backend plugin with pseudo-random inputs and prints the latency.
//   cargo run --example run_qnn -- <plugin> <sdk-root> <model> [npu|cpu|gpu] [runs] [cache-dir]
use std::time::Instant;

fn main() -> uairt::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: run_qnn <plugin> <sdk-root> <model> [npu|cpu|gpu] [runs] [cache-dir]");
        std::process::exit(2);
    }
    let device = args.get(4).map_or("npu", String::as_str);
    let runs: usize = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(20);

    // Without it the NPU libraries fall back to a path that crashed the process (heap corruption).
    if cfg!(windows) && device == "npu" && std::env::var_os("ADSP_LIBRARY_PATH").is_none() {
        eprintln!(r"set ADSP_LIBRARY_PATH to <QAIRT root>\lib\hexagon-v73\unsigned before running");
        std::process::exit(2);
    }
    uairt::load_backend_library(&args[1])?;
    let mut options = vec![("device", device), ("sdk_root", args[2].as_str())];
    if let Some(dir) = args.get(6) {
        options.push(("cache_dir", dir.as_str()));
    }

    let started = Instant::now();
    let engine = uairt::Engine::new("qnn", &options)?;
    let mut model = engine.load_model(&args[3])?;
    println!("init_ms={:.1}", started.elapsed().as_secs_f64() * 1e3);

    let mut seed = 1u32;
    let inputs: Vec<Vec<u8>> = model
        .inputs()
        .iter()
        .map(|info| {
            (0..info.nbytes())
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 24) as u8
                })
                .collect()
        })
        .collect();
    let mut outputs: Vec<Vec<u8>> = model.outputs().iter().map(|o| vec![0u8; o.nbytes()]).collect();
    for (i, info) in model.inputs().iter().enumerate() {
        println!("input {i}: {:?} {:?}", info.dtype, info.shape);
    }

    let input_refs: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
    let mut latencies = Vec::with_capacity(runs);
    for _ in 0..runs.max(1) {
        let mut output_refs: Vec<&mut [u8]> = outputs.iter_mut().map(Vec::as_mut_slice).collect();
        let start = Instant::now();
        model.run(&input_refs, &mut output_refs)?;
        latencies.push(start.elapsed().as_secs_f64() * 1e3);
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "latency_ms runs={} min={:.2} median={:.2}",
        latencies.len(),
        latencies[0],
        latencies[latencies.len() / 2]
    );
    for (i, output) in outputs.iter().enumerate() {
        let sum: u64 = output.iter().map(|&b| u64::from(b)).sum();
        println!("output {i}: {} bytes, byte sum {sum}", output.len());
    }
    Ok(())
}
