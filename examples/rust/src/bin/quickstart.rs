// SPDX-License-Identifier: Apache-2.0
// quickstart.c in Rust: the built-in "reference" backend copies one float32 [1, 4] input to the output.
use uairt::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("UAIRT {}", uairt::version());
    let engine = Engine::new("reference", &[])?;
    let mut model = engine.load_model("unused")?;
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut output = [0.0f32; 4];
    model.run_f32(&[&input], &mut [&mut output])?;
    println!("output: {} {} {} {}", output[0], output[1], output[2], output[3]);
    if output[3] == 4.0 { Ok(()) } else { Err("unexpected output".into()) }
}
