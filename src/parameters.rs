
use std::sync::atomic::{Ordering, AtomicU8};
use midir::{MidiInput, MidiInputConnection};

/// the parameters to be input to the shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Parameters {
  /// the three parts of the cross-color matrix: forward, self, and backward
  pub mat_parts: [f32; 3],
  pub step_factor: f32,
  pub fade_factor: f32,
  /// the standard deviations of the cellular automata response curves
  pub stdevs: [f32; 3],
  pub small_stdev: f32,
  pub big_stdev: f32,
}


/// the state of the control board, or 0xff if no change
pub type ControlState = [AtomicU8; 32];

pub fn new_control_state() -> ControlState {
  // TODO: find a less dumb way to do this
  unsafe { std::mem::transmute::<[u8; 32], [AtomicU8; 32]>([0xffu8; 32]) }
}

pub fn read_control_state(cs: &ControlState, i: usize) -> Option<u8>{
    let x = cs[i].swap(255, Ordering::Relaxed);
    if x == 255 {
        None
    } else {
        Some(x)
    }
}

impl Parameters {
  pub fn update_from_controller(&mut self, cs: &ControlState) {
    let f = |x: u8| 2.0 * (x as f32 / 127.0) - 1.0;
    for i in 0..3 {
        if let Some(x) = read_control_state(cs, i).map(|x| f(x)) {
            self.mat_parts[i] = x;
        }
    }
    let f = |x: u8| -> f32 {
        let y = x as f32 / 127.0;
        y * y
    };
    for i in 3..6 {
        if let Some(x) = read_control_state(cs, i).map(|x| f(x)) {
            self.stdevs[i - 3] = x;
        }
    }
    if let Some(x) = read_control_state(cs, 6).map(|x| f(x) * 0.2) {
        self.fade_factor = x;
    }
    if let Some(x) = read_control_state(cs, 7).map(|x| f(x) * 0.5) {
        self.step_factor = x;
    }
  }
}

pub fn init_control<'a>(control: std::sync::Arc<ControlState>) -> Result<MidiInputConnection<std::sync::Arc<ControlState>>, std::io::Error> {
  let midi_input = MidiInput::new("midic_test").map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "could not open MIDI driver"))?;
  let mut ports = midi_input.ports();
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
    if bytes.len() != 3 || bytes[0] != 0xb0 || bytes[1] >= 32 {
      return;
    }
    cs[bytes[1] as usize].store(bytes[2], Ordering::Relaxed);
  }, control).map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "could not connect to MIDI port"))?;
  Ok(conn)
}
