/// Bandpass filter to guide the random noise that drives the parameters. I
/// took most of this code from one of my other projects.

use num::Zero;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg};
use std::collections::VecDeque;
use std::f64::consts::PI;

pub trait Filter<F> {
  /// The number of samples of lag on the output
  fn latency(&self) -> usize;
  /// Put a sample into and get a sample out of the filter.
  fn pump(&mut self, samp: F) -> F;
}

/// A filter which filters signals through two filters (first the left, then
/// the right).
#[derive(Debug, Clone)]
pub struct Compose<F, Fl1: Filter<F>, Fl2: Filter<F>>(pub Fl1, pub Fl2, pub std::marker::PhantomData<F>);

impl<F, Fl1: Filter<F>, Fl2: Filter<F>> Compose<F, Fl1, Fl2> {
  pub fn new(fl1: Fl1, fl2: Fl2) -> Self {
    Self(fl1, fl2, std::marker::PhantomData {})
  }
}

impl<F, Fl1: Filter<F>, Fl2: Filter<F>> Filter<F> for Compose<F, Fl1, Fl2> {
  fn latency(&self) -> usize {
    self.0.latency() + self.1.latency()
  }

  fn pump(&mut self, samp: F) -> F {
    self.1.pump(self.0.pump(samp))
  }
}

/// FIR feedforward filter.
#[derive(Debug, Clone)]
pub struct Fir<F> {
  pub kernel: Vec<F>,
  samples: VecDeque<F>,
}

impl<F: Copy> Fir<F> {
  pub fn new(kernel: &[F]) -> Self {
    let len = kernel.len();
    Self {
      kernel: Vec::from(kernel),
      samples: VecDeque::with_capacity(len + 1),
    }
  }
}

impl<F: Zero + AddAssign + Neg<Output = F> + MulAssign + Copy> Fir<F> {
  pub fn kernel_from_zeros(zeros: &[F], scale: F) -> Vec<F> {
    let mut poly = poly_from_roots(zeros, scale);
    poly.reverse();
    poly
  }

  pub fn from_zeros(zeros: &[F], scale: F) -> Self {
    let kernel = Self::kernel_from_zeros(zeros, scale);
    let len = kernel.len();
    Self {
      kernel,
      samples: VecDeque::with_capacity(len),
    }
  }
}

fn poly_from_roots<F: Zero + Add<Output = F> + AddAssign + MulAssign + Neg<Output = F> + Copy>(roots: &[F], scale: F) -> Vec<F> {
    let mut poly = vec![F::zero(); roots.len() + 1];
    *poly.last_mut().unwrap() = scale;
    for (i, root) in roots.iter().cloned().enumerate() {
      for j in roots.len() - i - 1 .. roots.len() {
        poly[j] *= -root;
        let x = poly[j + 1];
        poly[j] += x;
      }
      *poly.last_mut().unwrap() *= -root;
    }
    poly
}

impl<F: Zero + Mul<Output = F> + AddAssign + Copy> Filter<F> for Fir<F> {
  fn latency(&self) -> usize { 0 }

  fn pump(&mut self, sample: F) -> F {
    self.samples.push_front(sample);
    if self.samples.len() > self.kernel.len() {
      self.samples.pop_back();
    }
    let mut filter_output = F::zero();
    for i in 0..self.samples.len() {
      filter_output += self.kernel[i] * self.samples[i];
    }
    filter_output
  }
}

/// Composition of single-pole IIR feedback filters.
#[derive(Debug, Clone)]
pub struct MultistageIir<F> {
  pub stages: Vec<IirStage<F>>,
  outputs: Vec<F>,
}

#[derive(Debug, Clone)]
pub enum IirStage<F> {
  Single(F),
  Pair(F, F),
}

impl<F> IirStage<F> {
  const fn length(&self) -> usize {
    match self {
      IirStage::Single(_) => 1,
      IirStage::Pair(_, _) => 2,
    }
  }
}

impl<F: Clone + Zero> MultistageIir<F> {
  pub fn from_stages(stages: &[IirStage<F>]) -> Self {
    Self {
      stages: Vec::from(stages),
      outputs: vec![F::zero(); stages.iter().map(|s| s.length()).sum::<usize>() + 1],
    }
  }
}

impl<F: Add<Output = F> + Mul<Output = F> + Copy> Filter<F> for MultistageIir<F> {
  fn latency(&self) -> usize { 0 }

  fn pump(&mut self, sample: F) -> F {
    self.outputs[0] = sample;
    let mut i = 0;
    for stage in self.stages.iter() {
      match stage {
        IirStage::Single(c) => {
          let next = self.outputs[i] + self.outputs[i + 1] * *c;
          self.outputs[i + 1] = next;
        }
        IirStage::Pair(c1, c2) => {
          let next = self.outputs[i] + self.outputs[i + 1] * *c1 + self.outputs[i + 2] * *c2;
          self.outputs[i + 1] = self.outputs[i + 2];
          self.outputs[i + 2] = next;
        }
      }
      i += stage.length();
    }
    self.outputs[self.outputs.len() - 1]
  }
}

type Real = f64;
type Complex = num::complex::Complex<f64>;

/// Butterworth filter poles in Laplace space
pub fn lowpass_s_pole(order: usize, i: usize) -> Complex {
  let arg = PI * (0.5 + (2 * i + 1) as Real / (2 * order) as Real);
  Complex::from_polar(4.0 / PI, arg)
}

pub fn bandpass_pole(order: usize, i: usize, lower: Real, upper: Real) -> Complex {
    let s_pole = lowpass_s_pole(order, i);
    let lowpass_cutoff = (upper - lower)  * 0.5;
    s_to_z_bilinear(s_pole * lowpass_cutoff + Complex::new(0., lowpass_cutoff + lower))
}

/// Convert the coordinates of a pole in Laplace space to a pole in Z-space
/// using the bilinear transform
pub fn s_to_z_bilinear(s: Complex) -> Complex {
  let spi = s * PI;
  (1. + spi) / (1. - spi)
}
