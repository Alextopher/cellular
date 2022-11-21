extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

mod gpu_gore;
mod parameters;

use gpu_gore::VulkanData;
use parameters::ControlState;

use image::RgbaImage;
use nokhwa::{
    pixel_format::RgbAFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution as CamRes},
    Camera,
};
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

const DEFAULT_DIMS: [u32; 2] = [100, 100];

struct VulkanWindow {
    sfc: Arc<Surface<Window>>,
    el: EventLoop<()>,
    vk: VulkanData<Window>,
    control_state: Arc<ControlState>,
}

impl VulkanWindow {
    fn init_winit(inst: &Arc<Instance>) -> (Arc<Surface<Window>>, EventLoop<()>) {
        let el = EventLoop::new();
        let sfc = WindowBuilder::new()
            .with_title("ultimate")
            .with_visible(true)
            .with_inner_size(Size::Physical(PhysicalSize {
                width: DEFAULT_DIMS[0],
                height: DEFAULT_DIMS[1],
            }))
            .build_vk_surface(&el, inst.clone())
            .expect("could not create Vulkan surface");
        (sfc, el)
    }

    fn init(control_state: Arc<ControlState>, res: CamRes) -> Self {
        let inst = VulkanData::<()>::init_vk_instance();
        let (sfc, el) = Self::init_winit(&inst);
        let mut vk = VulkanData::init(inst, sfc.clone(), res);
        vk.randomize_buffer(DEFAULT_DIMS);
        Self {
            sfc,
            el,
            vk,
            control_state,
        }
    }

    fn do_loop(self, get_frame: impl Fn() -> Option<RgbaImage> + 'static) {
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
                        (Some(VirtualKeyCode::Space), _) => {
                            let mut wl = vk.params_write_lock();
                            println!("parameter state: {:?}", *wl);
                            std::mem::drop(wl);
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
                    event: WindowEvent::CursorMoved { position: _pos, .. },
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
                    vk.randomize_buffer(dims);
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
                    if elapsed > last_report_elapsed + 5.0 {
                        println!("redrawing; fr: {fr:10.5}, elapsed: {elapsed:10.5}, frames since reset: {step_counter}");
                        last_report_elapsed = elapsed;
                    }
                    if let Some(frame) = get_frame() {
                        vk.capture_frame(frame);
                    }
                    let mut wl = vk.params_write_lock();
                    wl.update_from_controller(&*self.control_state);
                    std::mem::drop(wl);
                    vk.do_frame(&sfc, dims);
                }
                _ => {}
            }
        });
    }
}

fn main() {
    use std::sync::mpsc::TryRecvError;
    //println!(
    //    "layers: {:?}",
    //    vulkano::instance::layers_list()
    //        .expect("Unable to enumerate available Vulkan layers")
    //        .map(|ly| (ly.name().to_string(), ly.description().to_string()))
    //        .collect::<Vec<_>>()
    //);

    // TODO
    //nokhwa::nokhwa_initialize(something);
    let (tx, rx) = std::sync::mpsc::sync_channel(4);
    let mut cam = Camera::new(
        CameraIndex::Index(0),
        RequestedFormat::new::<RgbAFormat>(RequestedFormatType::None),
    )
    .expect("could not initialize camera!");
    cam.open_stream().expect("could not open camera stream");
    let res = cam.resolution();
    println!("camera frame rate: {}", cam.frame_rate());
    let _ = std::thread::spawn(move || loop {
        tx.send(
            cam.frame()
                .expect("could not capture camera frame!")
                .decode_image::<RgbAFormat>()
                .expect("could not convert camera frame to RGBA format!"),
        )
        .expect("could not send camera frame to GPU thread!");
        //println!("captured frame");
    });
    let control_state = Arc::new(parameters::new_control_state());
    let conn =
        parameters::init_control(control_state.clone()).expect("could not initialize controller!");
    let vw = VulkanWindow::init(control_state, res);
    vw.do_loop(move || -> Option<image::RgbaImage> {
        match rx.try_recv() {
            Ok(frame) => Some(frame),
            Err(TryRecvError::Disconnected) => panic!("second thread terminated unexpectedly"),
            Err(TryRecvError::Empty) => None,
        }
    });
    std::mem::drop(conn);
}
