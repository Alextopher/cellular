extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

mod gpu_gore;

use std::sync::Arc;
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::{PhysicalSize, Size},
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use gpu_gore::{VulkanData};

use rand::Rng;

const DEFAULT_DIMS: [u32; 2] = [100, 100];

struct VulkanWindow {
    sfc: Arc<Surface<Window>>,
    el: EventLoop<()>,
    vk: VulkanData<Window>,
}

impl VulkanWindow {
    fn init_winit(inst: &Arc<Instance>) -> (Arc<Surface<Window>>, EventLoop<()>) {
        let el = EventLoop::new();
        let sfc = WindowBuilder::new()
            .with_title("mesmerise")
            .with_visible(true)
            .with_inner_size(Size::Physical(PhysicalSize {
                width: DEFAULT_DIMS[0],
                height: DEFAULT_DIMS[1],
            }))
            .build_vk_surface(&el, inst.clone())
            .expect("could not create Vulkan surface");
        (sfc, el)
    }

    fn init() -> Self {
        let inst = VulkanData::<()>::init_vk_instance();
        let (sfc, el) = Self::init_winit(&inst);
        let vk = VulkanData::init(inst, sfc.clone());
        //vk.scramble(DEFAULT_DIMS);
        Self { sfc, el, vk }
    }

    fn do_loop(self) {
        let el = self.el;
        let sfc = self.sfc;
        let mut vk = self.vk;
        let begin_time = std::time::Instant::now();
        let mut last_time = begin_time;
        let mut frame_was_last_time = true;
        let mut step_counter = 0;
        let mut last_report_elapsed = 0.0;

        el.run(move |ev, _, elcf| {
            //println!("running event loop: {:?}", ev);
            *elcf = ControlFlow::Wait;
            match ev {
                Event::MainEventsCleared { .. } => {
                    if !frame_was_last_time {
                        last_time = std::time::Instant::now();
                    }
                    frame_was_last_time = false;
                    sfc.window().request_redraw();
                }
                Event::WindowEvent {
                    event:
                        WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: key_state,
                                    virtual_keycode: key,
                                    ..
                                },
                            ..
                        },
                    ..
                } => {
                    println!("key: {:?} {:?}", key_state, key);
                    match (key, key_state) {
                        (Some(VirtualKeyCode::Q), _) => {
                            *elcf = ControlFlow::Exit;
                        }
                        _ => (),
                    }
                }
                Event::WindowEvent {
                    event: WindowEvent::ModifiersChanged(mstate),
                    ..
                } => {
                    println!("new modifier set: {:?}", mstate);
                }
                Event::WindowEvent {
                    event: WindowEvent::CursorMoved { position: pos, .. },
                    ..
                } => {}
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    println!("closing window");
                    *elcf = ControlFlow::Exit;
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    println!("resizing to {}x{}", size.width, size.height);
                    let dims = [size.width, size.height];
                    vk.rebuild_swapchain(&sfc, dims);
                    vk.rebuild_storage(dims);
                    //vk.scramble(dims);
                    step_counter = 0;
                }
                Event::RedrawRequested(_) => {
                    let now = std::time::Instant::now();
                    let inter_frame_time = now.duration_since(last_time).as_secs_f32();
                    frame_was_last_time = true;
                    last_time = now;
                    let fr = inter_frame_time.recip();
                    let elapsed = now.duration_since(begin_time).as_secs_f32();

                    step_counter += 1;
                    let size = sfc.window().inner_size();
                    let dims = [size.width, size.height];
//                    let phase = step_counter as f32 * 0.001 * std::f32::consts::TAU;
//                    let warp = |x| 350.0 * (0.7 - f32::exp(1.0 * f32::sin(x)));
//                    let wide = warp(phase);
//                    let narrow = warp(phase - std::f32::consts::FRAC_PI_2);
                    if elapsed > last_report_elapsed + 2.0 {
                      println!("redrawing; fr: {fr:10.5}, elapsed: {elapsed:10.5}");
                      last_report_elapsed = elapsed;
                    }
                    vk.do_frame(&sfc, dims);
                }
                _ => {}
            }
        });
    }
}

fn main() {
    //println!(
    //    "layers: {:?}",
    //    vulkano::instance::layers_list()
    //        .expect("Unable to enumerate available Vulkan layers")
    //        .map(|ly| (ly.name().to_string(), ly.description().to_string()))
    //        .collect::<Vec<_>>()
    //);

    let vw = VulkanWindow::init();
    vw.do_loop();
}
