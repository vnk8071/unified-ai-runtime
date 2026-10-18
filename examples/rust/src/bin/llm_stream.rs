// SPDX-License-Identifier: Apache-2.0
// Streams text from a GGUF model through the llamacpp backend with uairt_llm sampling:
//   llm_stream <libuairt_backend_llamacpp> <model.gguf> "<prompt>" <max_tokens> [temperature [top_k [top_p [seed]]]]
use std::io::Write;
use uairt::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 || args.len() > 9 {
        eprintln!("usage: {} <plugin> <model.gguf> \"<prompt>\" <max_tokens> [temperature [top_k [top_p [seed]]]]", args[0]);
        std::process::exit(2);
    }
    let arg = |i: usize| args.get(i).map(String::as_str);
    uairt::load_backend_library(&args[1])?;
    let engine = Engine::new("llamacpp", &[("n_gpu_layers", "99")])?;
    let mut model = engine.load_model(&args[2])?;

    print!("{}", args[3]);
    let stdout = std::io::stdout();
    let (tokens, reason) = model.generate(
        &args[3],
        args[4].parse()?,
        arg(5).map_or(Ok(0.0), str::parse)?,
        arg(6).map_or(Ok(0), str::parse)?,
        arg(7).map_or(Ok(1.0), str::parse)?,
        arg(8).map_or(Ok(0), str::parse)?,
        |piece| {
            let mut out = stdout.lock();
            let _ = out.write_all(piece);
            let _ = out.flush();
            false
        },
    )?;
    println!();
    eprintln!("{tokens} tokens, finish_reason={reason}");
    Ok(())
}
