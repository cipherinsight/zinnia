// Risc 0 program code
use risc0_zkvm::guest::env;

fn main() {
    // read inputs
    let degree: f64 = env::read();
    let result: f64 = env::read();

    // π constant and degree→radian conversion
    let pi: f64 = 3.141592653589793;
    let rad: f64 = degree * pi / 180.0;
    let computed: f64 = rad.cos();

    // verify equality (with safe tolerance)
    assert!((result - computed).abs() < 1e-9);

    // env::commit(&output);
}
