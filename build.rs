use std::{
    env,
    fs::File,
    io::{Result, Write},
    path::Path,
};

// width of kernel is either 2 * NUM_STEPS + 1 or 2 * NUM_STEPS - 1
const NUM_STEPS: usize = 120;
// standard deviation of kernel
//const STDEV: f64 = 1.36;
const STDEV: f64 = 10.0;
// strings which appear in the output code
const VARNAME_NUM_STEPS: &str = "int num_steps";

fn main() -> Result<()> {
    eprintln!("NUM_STEPS: {NUM_STEPS}, STDEV: {STDEV}");
    let big_stdev = 30.0;
    let small_stdev = 15.0;
    let big_int = integral(big_stdev);
    let small_int = integral(small_stdev);
    // our kernel will be difference of Gaussians. in order to make the kernel
    // zero at the origins, the two Gaussians which are added must have unitary
    // scaling factors of different signs. the integral under the whole curve
    // will be one minus the other -- we must subtract the square differences to
    // get the inverse of the scaling factor for the 2D curve. therefore, for the
    // 1D curve, it is the square root of the quantity given previously
    // therefore, for the 1D curve, it is the square root of the quantity given
    // previously.
    let dog_int_2d = big_int * big_int - small_int * small_int;
    let sf = dog_int_2d.sqrt().recip();

    let out_path = Path::new("src").join("kernel.glsl");
    let mut f = File::create(out_path)?;

    #[rustfmt::skip]
    writeln!(f, "// auto-generated by build.rs -- coefficients for Gaussian convolution")?;
    writeln!(f, "const {VARNAME_NUM_STEPS} = {NUM_STEPS};")?;
    write_kernel(&mut f, big_stdev, "float big_stdev", "float big_weights", sf)?;
    write_kernel(&mut f, small_stdev, "float small_stdev",  "float small_weights", sf)?;
    // TODO: move this somewhere else :P
    write!(f, "
float kernel_weight(float stdev, int i) {{\n
  float arg = float(i) / stdev;
  float val = exp(-arg * arg);\n
  if (i == 0) {{\n
    return 0.5 * val;\n
  }} else {{\n
    return val;\n
  }}\n
}}\n
"
        )?;

    Ok(())
}

// integral under the (1D) Gaussian curve -- square for 2D
fn integral(stdev: f64) -> f64 {
    std::f64::consts::PI.sqrt() * stdev
}

fn write_kernel(
    f: &mut File,
    stdev: f64,
    varname_stdev: &str,
    varname_weights: &str,
    scaling_factor: f64,
) -> Result<()> {
    let stdev_recip = stdev.recip();
    let mut coeffs = vec![];
    // halved becuase it will be double-counted when the kernel is computed bidirectionally
    coeffs.push(0.5);
    // compute plain kernel, without sampler tricks
    for i in 1..NUM_STEPS {
        coeffs.push((-((i * i) as f64 * stdev_recip * stdev_recip)).exp());
    }
    eprintln!("coeffs: {coeffs:?}");
    let weights: Vec<_> = coeffs.into_iter().map(|x| x * scaling_factor).collect();
    eprintln!("weights: {weights:?}");

    // print the GLSL code
    writeln!(f, "const {varname_stdev} = {stdev};");
    writeln!(f, "const {varname_weights}[{NUM_STEPS}] = {{")?;
    for i in 0..NUM_STEPS {
        writeln!(f, "\t{:2.15},", weights[i])?;
    }
    writeln!(f, "}};")?;

    Ok(())
}
