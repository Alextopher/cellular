
use std::sync::atomic::{Ordering, AtomicU8};
use midir::{MidiInput, MidiInputConnection};

/// the parameters to be input to the shader
#[repr(C)]
pub struct Parameters {
  /// the three parts of the cross-color matrix: forward, self, and backward
  mat_parts: (f32, f32, f32),
  blend_factor: f32,
  fade_factor: f32,
  /// the standard deviations of the cellular automata response curves
  stdevs: (f32, f32, f32),
}


/// the state of the control board, or 0xff if no change
pub type ControlState = [AtomicU8; 32];

pub fn new_control_state() -> ControlState {
  // TODO: find a less dumb way to do this
  unsafe { std::mem::transmute::<[u8; 32], [AtomicU8; 32]>([0u8; 32]) }
}

impl Parameters {
  fn update_from_controller(&mut self, cs: &ControlState) {
    let r = |i: usize| cs[i].load(Ordering::Relaxed);
    let f = |x: u8| if x == 255 { None } else { Some(x as f32 / 127.0) };
    if let Some(x) = f(r(0)) {
      self.mat_parts.0 = x;
    }
    if let Some(x) = f(r(1)) {
      self.mat_parts.1 = x;
    }
    if let Some(x) = f(r(2)) {
      self.mat_parts.2 = x;
    }
  }
}

pub fn init_control<'a>(control: std::sync::Arc<ControlState>) -> Result<MidiInputConnection<std::sync::Arc<ControlState>>, std::io::Error> {
  let midi_input = MidiInput::new("midic_test").map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "could not open MIDI driver"))?;
  let mut ports = midi_input.ports();
  println!("MIDI input ports:");
  let mut idx = None;
  for (i, port) in ports.iter().enumerate() {
    let name = midi_input.port_name(port).unwrap_or("unknown".into());
    if name.contains("nanoKONTROL2") {
      idx = Some(i);
    }
  }
  let port;
  if let Some(i) = idx {
    port = ports.swap_remove(i);
  } else {
    return Err(std::io::Error::new(std::io::ErrorKind::Other, "could not locate the nanoKONTROL2 device"));
  }

  let conn = midi_input.connect(&port, "midic_test", |_, bytes, cs| {
    if bytes.len() != 3 || bytes[0] != 0xb0 || bytes[2] >= 32 {
      return;
    }
    cs[bytes[1] as usize].store(bytes[2], Ordering::Relaxed);
  }, control).map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "could not connect to MIDI port"))?;
  Ok(conn)
}