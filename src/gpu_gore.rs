/// A module for containing the ugly Vulkan code

mod scramble_shader {
    vulkano_shaders::shader! {
      ty: "compute",
      path: "src/scramble.glsl",
    }
}

mod xblur_shader {
    vulkano_shaders::shader! {
      ty: "compute",
      path: "src/xblur.glsl",
    }
}

mod drift_shader {
    vulkano_shaders::shader! {
      ty: "compute",
      path: "src/drift.glsl",
    }
}

use rand::Rng;
use std::collections::hash_set::HashSet;
use std::fmt::Debug;
use std::sync::Arc;
use super::DEFAULT_DIMS;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyImageInfo, PrimaryCommandBuffer};
use vulkano::descriptor_set::{
    layout::DescriptorSetLayout, persistent::PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::{
    physical::{PhysicalDevice, QueueFamily},
    Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo,
};
use vulkano::format::Format as ImageFormat;
use vulkano::image::{
    view::{ImageView, ImageViewCreateInfo},
    ImageAspects, ImageCreateFlags, ImageDimensions, ImageSubresourceRange, ImageUsage,
    StorageImage, SwapchainImage,
};
use vulkano::instance::{Instance, InstanceCreateInfo, Version as VkVersion};
use vulkano::pipeline::{compute::ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sampler::{Sampler, SamplerCreateInfo};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    self, CompositeAlpha, Surface, SurfaceCapabilities, SurfaceInfo, Swapchain, SwapchainCreateInfo,
};
use vulkano::sync::{GpuFuture, Sharing};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize, Size},
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[repr(C)]
pub struct ScrambleParams {
    prng_seed: u32,
}

#[repr(C)]
pub struct DriftParams {
    prng_seed: u32,
    flags: u32,
    mult1: f32,
    mult2: f32,
    invocation_size: [i32; 2],
}


pub struct VulkanData<W> {
    inst: Arc<Instance>,
    device: Arc<Device>,
    compute_queue: Arc<Queue>,
    present_queue: Arc<Queue>,
    scramble: ShaderModData,
    drift: ShaderModData,
    xblur: ShaderModData,
    arena_image: Arc<StorageImage>,
    xblur_image: Arc<StorageImage>,
    swapchain: Arc<Swapchain<W>>,
    swapchain_images: Vec<Arc<SwapchainImage<W>>>,
    scramble_desc_set: Arc<PersistentDescriptorSet>,
    drift_desc_set: Arc<PersistentDescriptorSet>,
    xblur_desc_set: Arc<PersistentDescriptorSet>,
    frame_future: Option<Box<dyn GpuFuture>>,
}

struct ShaderModData {
  pipeline: Arc<ComputePipeline>,
  desc_set_layout: Arc<DescriptorSetLayout>,
}

#[derive(Debug)]
struct QueueFamilies {
    compute_idx: Option<usize>,
    present_idx: Option<usize>,
}

impl QueueFamilies {
    fn new() -> Self {
        QueueFamilies {
            compute_idx: None,
            present_idx: None,
        }
    }

    fn is_complete(&self) -> bool {
        self.compute_idx.is_some() && self.present_idx.is_some()
    }

    fn populate<W>(&mut self, dev: &PhysicalDevice, surface: &Arc<Surface<W>>) {
        for (i, qf) in dev.queue_families().enumerate() {
            if qf.supports_compute() {
                self.compute_idx = Some(i);
            }
            if qf.supports_surface(surface).unwrap() {
                self.present_idx = Some(i);
            }
        }
    }
}

const VALIDATION_LAYERS: &[&str] = &[
    //    "VK_LAYER_LUNARG_core_validation",
    //    "VK_LAYER_KHRONOS_validation",
    //    "VK_LAYER_LUNARG_standard_validation",
    //    "VK_LAYER_LUNARG_parameter_validation",
];
const USE_VALIDATION_LAYERS: bool = true;

impl<W: 'static + Debug + Sync + Send> VulkanData<W> {
    pub fn init(inst: Arc<Instance>, sfc: Arc<Surface<W>>) -> Self {
        use std::io::Write;
        let device_idxs = PhysicalDevice::enumerate(&inst)
            .enumerate()
            .filter_map(|(i, dev)| {
                if Self::check_physical_device(&dev, &sfc) {
                    Some((i, dev))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if device_idxs.is_empty() {
            panic!("No graphics devices with Vulkan support were detected!");
        }
        let device_idx = if device_idxs.len() == 1 {
            device_idxs[0].0
        } else {
            println!("Please select a device:");
            device_idxs.iter().enumerate().for_each(|(i, (_j, dev))| {
                println!("{}: {}", i, dev.properties().device_name);
            });
            print!("> ");
            std::io::stdout().flush().unwrap_or_else(|_| {
                println!("");
                ()
            });
            loop {
                let mut response = String::new();
                std::io::stdin().read_line(&mut response).unwrap();
                let len = response.len();
                if len == 0 {
                    panic!("Unexpected EOF while asking about which graphics device to use");
                }
                match (&response[0..len - 1]).parse::<usize>() {
                    Ok(n) => {
                        if n >= device_idxs.len() {
                            println!("That number isn't in the correct range. Please try again.");
                            print!("> ");
                            std::io::stdout().flush().unwrap();
                        } else {
                            break n;
                        }
                    }
                    Err(_) => {
                        println!("That doesn't appear to be a valid number. Please try again.");
                        print!("> ");
                        std::io::stdout().flush().unwrap();
                    }
                }
            }
        };
        let phy_dev = PhysicalDevice::from_index(&inst, device_idx).unwrap();
        let swp_caps = phy_dev
            .surface_capabilities(sfc.as_ref(), SurfaceInfo::default())
            .expect("unable to determine capabilities of surface with device!");
        let (device, _qis, compute_queue, present_queue) = Self::create_device(&sfc, phy_dev);
        let scramble = Self::make_pipeline(&device, scramble_shader::load(device.clone()).expect("could not load scramble shader module!"));
        let drift = Self::make_pipeline(&device, drift_shader::load(device.clone()).expect("could not load drift shader module!"));
        let xblur = Self::make_pipeline(&device, xblur_shader::load(device.clone()).expect("could not load xblur shader module!"));
        let (swapchain, swapchain_images) = Self::make_swapchain(
            &device,
            &sfc,
            &swp_caps,
            &compute_queue,
            &present_queue,
            DEFAULT_DIMS,
        );
        let image_format = swapchain.image_format();
        let (arena_image, xblur_image) = Self::make_storage(
            device.clone(),
            compute_queue.family(),
            DEFAULT_DIMS,
            image_format,
        );
        let scramble_desc_set = Self::make_desc_set(
          arena_image.clone(),
          scramble.desc_set_layout.clone(),
          image_format,
          );
        let drift_desc_set = Self::make_desc_set(
          arena_image.clone(),
          drift.desc_set_layout.clone(),
          image_format,
          );
        let xblur_desc_set = Self::make_xblur_desc_set(
          device.clone(),
          arena_image.clone(),
          xblur_image.clone(),
          xblur.desc_set_layout.clone(),
          image_format,
          );
        Self {
            inst,
            device,
            compute_queue,
            present_queue,
            scramble,
            drift,
            xblur,
            arena_image,
            xblur_image,
            swapchain,
            swapchain_images,
            scramble_desc_set,
            drift_desc_set,
            xblur_desc_set,
            frame_future: None,
        }
    }

    pub fn init_vk_instance() -> Arc<Instance> {
        //let sup_ext = InstanceExtensions::supported_by_core()
        //    .expect("Unable to retrieve supported Vulkan extensions");
        //println!("supported Vulkan extensions: {:?}", sup_ext);

        let mut info = InstanceCreateInfo::default();
        info.enabled_extensions = vulkano_win::required_extensions();
        info.application_name = Some("mesmerise".into());
        info.application_version = VkVersion {
            major: 0,
            minor: 0,
            patch: 0,
        };

        if USE_VALIDATION_LAYERS {
            info.enabled_layers = VALIDATION_LAYERS.iter().map(|s| s.to_string()).collect();
        }
        Instance::new(info).expect("could not create Vulkan instance!")
    }


    const fn required_device_extensions() -> DeviceExtensions {
        DeviceExtensions {
            khr_swapchain: true,
            khr_swapchain_mutable_format: true,
            ..DeviceExtensions::none()
        }
    }

    const fn required_device_features() -> Features {
        Features {
            shader_int16: true,
            shader_int64: true,
            .. Features::none()
        }
    }

    fn check_physical_device(device: &PhysicalDevice, surface: &Arc<Surface<W>>) -> bool {
        let mut qfs = QueueFamilies::new();
        qfs.populate(device, surface);
        let sup_ext = device.supported_extensions();
        let sup_feats = device.supported_features();
        qfs.is_complete()
            && Self::required_device_extensions().difference(&sup_ext) == DeviceExtensions::none()
            && Self::required_device_features().difference(&sup_feats) == Features::none()
    }

    fn create_device(
        sfc: &Arc<Surface<W>>,
        phy_dev: PhysicalDevice,
    ) -> (Arc<Device>, QueueFamilies, Arc<Queue>, Arc<Queue>) {
        let mut qis = QueueFamilies::new();
        qis.populate(&phy_dev, &sfc);
        let qfs = vec![qis.compute_idx.unwrap(), qis.present_idx.unwrap()]
            .into_iter()
            .collect::<HashSet<usize>>()
            .into_iter()
            .map(|i| QueueCreateInfo::family(phy_dev.queue_family_by_id(i as u32).unwrap()))
            .collect::<Vec<_>>();
        let dci = DeviceCreateInfo {
            enabled_extensions: Self::required_device_extensions(),
            enabled_features: Self::required_device_features(),
            queue_create_infos: qfs,
            ..Default::default()
        };
        let (device, mut queue_iter) =
            Device::new(phy_dev, dci).expect("could not create graphics device");
        let compute_queue = queue_iter.next().unwrap();
        let present_queue = queue_iter.next().unwrap_or_else(|| compute_queue.clone());
        (device, qis, compute_queue, present_queue)
    }

    /// make pipeline and descriptor set layout for a shader
    fn make_pipeline(device: &Arc<Device>, shader_module: Arc<ShaderModule>) -> ShaderModData {
        let mut desc_set_layout = None;
        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            shader_module
                .entry_point("main")
                .expect("shader does not have an entry point!"),
            &(),
            None,
            |dsl| {
                desc_set_layout = Some(
                    DescriptorSetLayout::new(device.clone(), dsl[0].clone())
                        .expect("could not create descriptor set layout"),
                );
            },
        )
        .expect("could not create compute pipeline");
        ShaderModData {
            pipeline: compute_pipeline,
            desc_set_layout: desc_set_layout.expect("could not get the descriptor set layout of the pipeline!"),
        }
    }

    fn make_swapchain(
        device: &Arc<Device>,
        surface: &Arc<Surface<W>>,
        swp_cap: &SurfaceCapabilities,
        cmpq: &Arc<Queue>,
        pstq: &Arc<Queue>,
        dims: [u32; 2],
    ) -> (Arc<Swapchain<W>>, Vec<Arc<SwapchainImage<W>>>) {
        let sharing = {
            let cmpqfid = cmpq.family().id();
            let pstqfid = pstq.family().id();
            if cmpqfid == pstqfid {
                Sharing::Exclusive
            } else {
                Sharing::Concurrent(vec![cmpqfid, pstqfid].into())
            }
        };
        let calpha = {
            let supported = swp_cap.supported_composite_alpha;
            if supported.supports(CompositeAlpha::PostMultiplied) {
                println!("postmultiplied compositing enabled");
                CompositeAlpha::PostMultiplied
            } else if supported.supports(CompositeAlpha::Opaque) {
                println!("opaque compositing enabled");
                CompositeAlpha::Opaque
            } else {
                panic!("no valid modes of alpha compositing are supported");
            }
        };
        let usage = ImageUsage {
            storage: true,
            color_attachment: true,
            transfer_dst: true,
            ..ImageUsage::none()
        };
        let dims = swp_cap.current_extent.unwrap_or(dims);
        let supp_fmts = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .expect("could not query supported surface formats!");
        println!("supported formats: {supp_fmts:?}");
        let sci = SwapchainCreateInfo {
            min_image_count: u32::max(2, swp_cap.min_image_count),
            image_extent: dims,
            pre_transform: swp_cap.current_transform,
            composite_alpha: calpha,
            image_usage: usage,
            image_sharing: sharing,
            // to match the GLSL
            image_format: Some(ImageFormat::R8G8B8A8_UNORM),
            // TODO: select image_color_space more intelligently
            ..Default::default()
        };
        let (swapchain, swapchain_images) = Swapchain::new(device.clone(), surface.clone(), sci)
            .expect("unable to create swapchain!");
        let fmt = swapchain.image_format();
        println!("swapchain image format: {:?}", fmt);
        (swapchain, swapchain_images)
    }

    // make the GPU buffers: the arena image and the intermittent blur buffer
    fn make_storage(
        device: Arc<Device>,
        qf: QueueFamily,
        dims: [u32; 2],
        format: ImageFormat,
    ) -> (Arc<StorageImage>, Arc<StorageImage>) {
        let usage = ImageUsage {
            transfer_src: true,
            sampled: true,
            storage: true,
            ..ImageUsage::none()
        };
        let dimensions = ImageDimensions::Dim2d {
            width: dims[0],
            height: dims[1],
            array_layers: 1,
        };
        let arena_image = StorageImage::with_usage(
            device.clone(),
            dimensions,
            format,
            usage,
            ImageCreateFlags::none(),
            [qf],
        )
        .expect("could not create arena image!");
        let blur_image = StorageImage::with_usage(
            device,
            dimensions,
            format,
            usage,
            ImageCreateFlags::none(),
            [qf],
        )
        .expect("could not create blur image!");
        (arena_image, blur_image)
    }

    fn make_desc_set(arena_image: Arc<StorageImage>, desc_set_layout: Arc<DescriptorSetLayout>, format: ImageFormat) -> Arc<PersistentDescriptorSet> {
        let ivci = ImageViewCreateInfo {
            format: Some(format),
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects {
                    color: true,
                    ..ImageAspects::none()
                },
                mip_levels: 0..1,
                array_layers: 0..1,
            },
            ..Default::default()
        };
        let desc_set = PersistentDescriptorSet::new(
            desc_set_layout,
            [WriteDescriptorSet::image_view(
                0,
                ImageView::new(arena_image, ivci)
                    .expect("could not create image view for arena image"),
            )],
        )
        .expect("could not create persistent descriptor set for arena image");
        desc_set
    }

    fn make_xblur_desc_set(device: Arc<Device>, arena_image: Arc<StorageImage>,
        xblur_image: Arc<StorageImage>, desc_set_layout:
        Arc<DescriptorSetLayout>, format: ImageFormat) ->
        Arc<PersistentDescriptorSet> {
        let ivci = || ImageViewCreateInfo {
            format: Some(format),
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects {
                    color: true,
                    ..ImageAspects::none()
                },
                mip_levels: 0..1,
                array_layers: 0..1,
            },
            ..Default::default()
        };
        let desc_set = PersistentDescriptorSet::new(
            desc_set_layout,
            [
            WriteDescriptorSet::image_view_sampler(
                0,
                ImageView::new(arena_image, ivci())
                    .expect("could not create image view for arena image"),
                Sampler::new(device, SamplerCreateInfo::simple_repeat_linear_no_mipmap()).expect("could not create simple repeating linear sampler for an image on the chosen GPU!"),
            ),
            WriteDescriptorSet::image_view(
                1,
                ImageView::new(xblur_image, ivci())
                    .expect("could not create image view for arena image"),
            ),
            ],
        )
        .expect("could not create persistent descriptor set for arena image");
        desc_set
    }

    // rebuild the swapchain, in case the current one has become invalid (e.g.
    // due to size change)
    pub fn rebuild_swapchain(&mut self, sfc: &Arc<Surface<W>>, dims: [u32; 2]) {
        let phy_dev = self.device.physical_device();
        let swp_caps = phy_dev
            .surface_capabilities(sfc, SurfaceInfo::default())
            .unwrap();
        let dims = swp_caps.current_extent.unwrap_or(dims);
        let (swapchain, swapchain_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: dims,
                ..self.swapchain.create_info()
            })
            .expect("could not rebuild swapchain!");
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
    }

    // rebuild all of the GPU buffers, in case e.g. their size has changed
    pub fn rebuild_storage(&mut self, dims: [u32; 2]) {
        let fmt = self.swapchain.image_format();
        (self.arena_image, self.xblur_image) = Self::make_storage(
            self.device.clone(),
            self.compute_queue.family(),
            dims,
            fmt,
        );
        self.scramble_desc_set = Self::make_desc_set(
          self.arena_image.clone(),
          self.scramble.desc_set_layout.clone(),
          fmt,
          );
        self.drift_desc_set = Self::make_desc_set(
          self.arena_image.clone(),
          self.drift.desc_set_layout.clone(),
          fmt,
          );
        self.xblur_desc_set = Self::make_xblur_desc_set(
          self.device.clone(),
          self.arena_image.clone(),
          self.xblur_image.clone(),
          self.xblur.desc_set_layout.clone(),
          fmt,
          );
    }

    pub fn scramble(&mut self, dims: [u32; 2]) {
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.compute_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
            .expect("could not make command buffer builder!");
        let prng_seed: u32 = rand::thread_rng().gen();
        command_buffer_builder
            .bind_pipeline_compute(self.scramble.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.scramble.pipeline.layout().clone(),
                0,
                self.scramble_desc_set.clone(),
            )
            .push_constants(
                self.scramble.pipeline.layout().clone(),
                0,
                ScrambleParams { prng_seed },
            )
            .dispatch([(dims[0] + 31) / 32, (dims[1] + 31) / 32, 1])
            .expect("could not dispatch compute operation!");
        let command_buffer = command_buffer_builder
            .build()
            .expect("could not build command buffer!");
        let scramble_future = if let Some(fut) = self.frame_future.take() {
            fut.then_execute(self.compute_queue.clone(), command_buffer).expect("could not queue execution of scramble after current presentation!").boxed()
        } else {
            command_buffer.execute(self.compute_queue.clone()).expect("could not queue execution of scramble!").boxed()
        };
        scramble_future.flush().expect("could not flush command queue for scrambling!");
        self.frame_future = Some(scramble_future);
    }

    pub fn do_frame(&mut self, sfc: &Arc<Surface<W>>, dims: [u32; 2], mult1: f32, mult2: f32) {
        if let Some(mut last_frame) = self.frame_future.take() {
            last_frame.cleanup_finished();
        }
        let invocation_size = [(dims[0] as i32 + 1) / 2, (dims[1] as i32 + 1) / 2];
        loop {
            let (image_idx, suboptimal, acquire_future) =
                swapchain::acquire_next_image(self.swapchain.clone(), None)
                    .expect("could not acquire swapchain image!");
            let make_cmd_buffer = |flags: u32, mult1: f32, mult2: f32| {
              let mut builder = AutoCommandBufferBuilder::primary(
                self.device.clone(),
                self.compute_queue.family(),
                CommandBufferUsage::OneTimeSubmit,
              )
                .expect("could not make command buffer builder!");
                let prng_seed: u32 = rand::thread_rng().gen();
                builder
                  .bind_pipeline_compute(self.drift.pipeline.clone())
                  .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.drift.pipeline.layout().clone(),
                    0,
                    self.drift_desc_set.clone(),
                  )
                  .push_constants(
                    self.drift.pipeline.layout().clone(),
                    0,
                    DriftParams { prng_seed, flags, mult1, mult2, invocation_size },
                  )
                  .dispatch([(invocation_size[0] as u32 + 31) / 32, (invocation_size[1] as u32 + 31) / 32, 1])
                  .expect("could not dispatch compute operation!");
                let buffer = builder
                  .build()
                  .expect("could not build command buffer!");
                buffer
                };
            let even_buffer = make_cmd_buffer(0, -mult1, -mult2);
            let odd_buffer = make_cmd_buffer(1, -mult1, -mult2);
            let xblur_cmd_buffer = {
                let mut builder = AutoCommandBufferBuilder::primary(
                    self.device.clone(),
                    self.compute_queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .expect("could not make command buffer builder!");
                builder
                    .bind_pipeline_compute(self.xblur.pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        self.xblur.pipeline.layout().clone(),
                        0,
                        self.xblur_desc_set.clone(),
                    )
                    .dispatch([(dims[0] + 31) / 32, (dims[1] + 31) / 32, 1])
                    .expect("could not dispatch compute operation!")
                    .copy_image(CopyImageInfo::images(
                            self.xblur_image.clone(),
                            self.swapchain_images[image_idx].clone(),
                            ))
                  .expect("could not copy blur image to swapchain image!");
                builder.build().expect("could not build blur command buffer!")
            };
            let frame_future = acquire_future
                .then_execute(self.compute_queue.clone(), even_buffer)
                .expect("could not queue execution of command buffer!")
                .then_execute(self.compute_queue.clone(), odd_buffer)
                .expect("could not queue execution of command buffer!")
                .then_execute(self.compute_queue.clone(), xblur_cmd_buffer)
                .expect("could not queue blur operation!")
                .then_signal_semaphore()
                .then_swapchain_present(
                    self.present_queue.clone(),
                    self.swapchain.clone(),
                    image_idx,
                );
            let result = frame_future.flush();
            if let Err(vulkano::sync::FlushError::OutOfDate) = result {
                self.rebuild_swapchain(sfc, dims);
                continue;
            }
            result.expect("could not flush gpu queue!");
            self.frame_future = Some(frame_future.boxed());
            if suboptimal {
                self.rebuild_swapchain(sfc, dims);
            }
            break;
        }
    }
}
