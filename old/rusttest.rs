fn main() {
  println!("hello from rust!");

  const PARAMS: [f64; 3] = [10.0, 2.0, 1.0];
  const STEP: f64 = 10.0;
  const RANGE: f64 = 1000.0;

  let mut accumulator = 0.0f64;

  use std::time::Instant;
  let now = Instant::now();

  let (mut x, mut y) = (-RANGE, -RANGE);
  while x <= RANGE {
    while y <= RANGE {
      let coords = &[x, y];
      let result = calc_h10(coords, &PARAMS);
      accumulator += result;
      y += STEP;
    }
    x += STEP;
  }

  let elapsed = now.elapsed();
  println!("Elapsed: {elapsed:?}");
  println!("Answer: {accumulator:?}");
}

fn calc_h10(coords: &[f64; 2], params: &[f64; 3]) -> f64 {
  let φ = coords[0];
  let ψ = coords[1];

  let m = params[0];
  let φ0 = params[1];
  let ψ0 = params[2];

  2.0*ψ0*(φ - φ0)*(2.0*m.powi(4)*φ.powi(2)*(φ - φ0)*(φ*(φ.powi(2) + 1.0) + 2.0*φ - 2.0*φ0 - (φ - φ0)*(φ.powi(2) + 1.0)) - ψ0.powi(2)*(φ.powi(2) + 1.0).powi(2))/(φ.powi(2)*(4.0*m.powi(4)*φ.powi(2)*(φ - φ0).powi(2) + ψ0.powi(2)*(φ.powi(2) + 1.0).powi(2))*(m.powi(2)*(φ - φ0).powi(2) - 2.0*ψ*ψ0)*(φ - φ0).abs())
}