/// A module for containing the ugly Vulkan code

mod xblur_shader {
    vulkano_shaders::shader! {
      ty: "compute",
      path: "src/xblur.glsl",
    }
}

mod cells_shader {
    vulkano_shaders::shader! {
      ty: "compute",
      path: "src/cells.glsl",
    }
}

mod blit_shader {
    vulkano_shaders::shader! {
      ty: "compute",
      path: "src/blit.glsl",
    }
}

use super::DEFAULT_DIMS;
use rand::Rng;
use std::collections::hash_set::HashSet;
use std::fmt::Debug;
use std::sync::Arc;
use vulkano::buffer::{cpu_access::CpuAccessibleBuffer, BufferUsage, DeviceLocalBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfoTyped};
use vulkano::descriptor_set::{
    layout::DescriptorSetLayout, persistent::PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::{
    physical::{PhysicalDevice},
    Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo,
};
use vulkano::format::Format as ImageFormat;
use vulkano::image::{
    view::{ImageView, ImageViewCreateInfo},
    ImageAspects, ImageSubresourceRange, ImageUsage, SwapchainImage,
};
use vulkano::instance::{Instance, InstanceCreateInfo, Version as VkVersion};
use vulkano::library::VulkanLibrary;
use vulkano::pipeline::{compute::ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    self, CompositeAlpha, Surface, SurfaceCapabilities, SurfaceInfo, Swapchain, SwapchainCreateInfo, PresentInfo,
};
use vulkano::sync::{GpuFuture, Sharing};
use vulkano::DeviceSize;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BufferReadParams {
    height: i32,
    width: i32,
}

pub struct VulkanData<W> {
    inst: Arc<Instance>,
    device: Arc<Device>,
    compute_queue: Arc<Queue>,
    present_queue: Arc<Queue>,
    cells: ShaderModData,
    xblur: ShaderModData,
    blit: ShaderModData,
    arena_buffer: Arc<DeviceLocalBuffer<[f32]>>,
    xblur_buffer: Arc<DeviceLocalBuffer<[f32]>>,
    swapchain: Arc<Swapchain<W>>,
    swapchain_images: Vec<Arc<SwapchainImage<W>>>,
    cells_desc_set: Arc<PersistentDescriptorSet>,
    xblur_desc_set: Arc<PersistentDescriptorSet>,
    blit_desc_sets: Vec<Arc<PersistentDescriptorSet>>,
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
        for (i, qf) in dev.queue_family_properties().into_iter().enumerate() {
            if qf.queue_flags.compute {
                self.compute_idx = Some(i);
            }
            if dev.surface_support(i as u32, surface).unwrap() {
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
        let device_idxs = inst.enumerate_physical_devices()
            .expect("could not enumerate devices!")
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
        let phy_dev = inst.enumerate_physical_devices().unwrap().nth(device_idx).unwrap();
        let swp_caps = phy_dev
            .surface_capabilities(sfc.as_ref(), SurfaceInfo::default())
            .expect("unable to determine capabilities of surface with device!");
        let (device, _qis, compute_queue, present_queue) = Self::create_device(&sfc, phy_dev);
        let cells = Self::make_pipeline(
            &device,
            cells_shader::load(device.clone()).expect("could not load cells shader module!"),
        );
        let xblur = Self::make_pipeline(
            &device,
            xblur_shader::load(device.clone()).expect("could not load xblur shader module!"),
        );
        let blit = Self::make_pipeline(
            &device,
            blit_shader::load(device.clone()).expect("could not load blit shader module!"),
        );
        let (swapchain, swapchain_images) = Self::make_swapchain(
            &device,
            &sfc,
            &swp_caps,
            &compute_queue,
            &present_queue,
            DEFAULT_DIMS,
        );
        let image_format = swapchain.image_format();
        let (arena_buffer, xblur_buffer) = Self::make_storage(
            device.clone(),
            compute_queue.queue_family_index(),
            DEFAULT_DIMS,
        );
        let cells_desc_set = Self::make_cells_desc_set(
            device.clone(),
            arena_buffer.clone(),
            xblur_buffer.clone(),
            cells.desc_set_layout.clone(),
        );
        let xblur_desc_set = Self::make_xblur_desc_set(
            device.clone(),
            arena_buffer.clone(),
            xblur_buffer.clone(),
            xblur.desc_set_layout.clone(),
        );
        let blit_desc_sets = swapchain_images
            .iter()
            .cloned()
            .map(|si| {
                Self::make_blit_desc_set(
                    device.clone(),
                    arena_buffer.clone(),
                    si,
                    blit.desc_set_layout.clone(),
                    image_format,
                )
            })
            .collect();
        Self {
            inst,
            device,
            compute_queue,
            present_queue,
            cells,
            xblur,
            blit,
            arena_buffer,
            xblur_buffer,
            swapchain,
            swapchain_images,
            cells_desc_set,
            xblur_desc_set,
            blit_desc_sets,
            frame_future: None,
        }
    }

    pub fn init_vk_instance() -> Arc<Instance> {
        //let sup_ext = InstanceExtensions::supported_by_core()
        //    .expect("Unable to retrieve supported Vulkan extensions");
        //println!("supported Vulkan extensions: {:?}", sup_ext);

        let lib = VulkanLibrary::new().expect("no Vulkan library found!");
        let mut info = InstanceCreateInfo {
            enabled_extensions: vulkano_win::required_extensions(&lib),
            application_name: Some("ultimate".into()),
            application_version: VkVersion {
                major: 0,
                minor: 0,
                patch: 0,
            },
        .. Default::default()
        };

        if USE_VALIDATION_LAYERS {
            info.enabled_layers = VALIDATION_LAYERS.iter().map(|s| s.to_string()).collect();
        }
        Instance::new(lib, info).expect("could not create Vulkan instance!")
    }

    const fn required_device_extensions() -> DeviceExtensions {
        DeviceExtensions {
            khr_swapchain: true,
            khr_swapchain_mutable_format: true,
            ..DeviceExtensions::empty()
        }
    }

    const fn required_device_features() -> Features {
        Features {
            shader_int16: true,
            shader_int64: true,
            .. Features::empty()
        }
    }

    fn check_physical_device(device: &PhysicalDevice, surface: &Arc<Surface<W>>) -> bool {
        let mut qfs = QueueFamilies::new();
        qfs.populate(device, surface);
        let sup_ext = device.supported_extensions();
        let sup_feats = device.supported_features();
        qfs.is_complete()
            && Self::required_device_extensions().difference(&sup_ext) == DeviceExtensions::empty()
            && Self::required_device_features().difference(&sup_feats) == Features::empty()
    }

    fn create_device(
        sfc: &Arc<Surface<W>>,
        phy_dev: Arc<PhysicalDevice>,
    ) -> (Arc<Device>, QueueFamilies, Arc<Queue>, Arc<Queue>) {
        let mut qis = QueueFamilies::new();
        qis.populate(&phy_dev, &sfc);
        let qfs = vec![qis.compute_idx.unwrap(), qis.present_idx.unwrap()]
            .into_iter()
            .collect::<HashSet<usize>>()
            .into_iter()
            .map(|i| QueueCreateInfo { queue_family_index: i as u32, .. Default::default() })
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
            desc_set_layout: desc_set_layout
                .expect("could not get the descriptor set layout of the pipeline!"),
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
            let cmpqfid = cmpq.queue_family_index();
            let pstqfid = pstq.queue_family_index();
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
            ..ImageUsage::empty()
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
        qf: u32,
        dims: [u32; 2],
    ) -> (Arc<DeviceLocalBuffer<[f32]>>, Arc<DeviceLocalBuffer<[f32]>>) {
        let usage = BufferUsage {
            storage_buffer: true,
            transfer_dst: true,
            .. BufferUsage::empty()
        };
        let mkbuf = |factor| {
            DeviceLocalBuffer::array(
                device.clone(),
                (dims[0] * dims[1] * factor) as DeviceSize,
                usage,
                [qf],
            )
            .expect("could not create a GPU buffer!")
        };
        (mkbuf(4), mkbuf(8))
    }

    fn make_cells_desc_set(
        device: Arc<Device>,
        arena_buffer: Arc<DeviceLocalBuffer<[f32]>>,
        xblur_buffer: Arc<DeviceLocalBuffer<[f32]>>,
        desc_set_layout: Arc<DescriptorSetLayout>,
    ) -> Arc<PersistentDescriptorSet> {
        let desc_set = PersistentDescriptorSet::new(
            desc_set_layout,
            [
                WriteDescriptorSet::buffer(0, arena_buffer.clone()),
                WriteDescriptorSet::buffer(1, xblur_buffer.clone()),
            ],
        )
        .expect("could not create persistent descriptor set for arena image");
        desc_set
    }

    fn make_xblur_desc_set(
        device: Arc<Device>,
        arena_buffer: Arc<DeviceLocalBuffer<[f32]>>,
        xblur_buffer: Arc<DeviceLocalBuffer<[f32]>>,
        desc_set_layout: Arc<DescriptorSetLayout>,
    ) -> Arc<PersistentDescriptorSet> {
        let desc_set = PersistentDescriptorSet::new(
            desc_set_layout,
            [
                WriteDescriptorSet::buffer(0, arena_buffer.clone()),
                WriteDescriptorSet::buffer(1, xblur_buffer.clone()),
            ],
        )
        .expect("could not create persistent descriptor set for arena image");
        desc_set
    }

    fn make_blit_desc_set(
        device: Arc<Device>,
        arena_buffer: Arc<DeviceLocalBuffer<[f32]>>,
        swapchain_image: Arc<SwapchainImage<W>>,
        desc_set_layout: Arc<DescriptorSetLayout>,
        format: ImageFormat,
    ) -> Arc<PersistentDescriptorSet> {
        let ivci = ImageViewCreateInfo {
            format: Some(format),
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects {
                    color: true,
                    .. ImageAspects::empty()
                },
                mip_levels: 0..1,
                array_layers: 0..1,
            },
            ..Default::default()
        };
        let desc_set = PersistentDescriptorSet::new(
            desc_set_layout,
            [
                WriteDescriptorSet::image_view(
                    0,
                    ImageView::new(swapchain_image, ivci)
                        .expect("could not create image view for swapchain image"),
                ),
                WriteDescriptorSet::buffer(1, arena_buffer.clone()),
            ],
        )
        .expect("could not create persistent descriptor set for blit shader");
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
        (self.arena_buffer, self.xblur_buffer) = Self::make_storage(
            self.device.clone(),
            self.compute_queue.queue_family_index(),
            dims,
        );
        self.cells_desc_set = Self::make_cells_desc_set(
            self.device.clone(),
            self.arena_buffer.clone(),
            self.xblur_buffer.clone(),
            self.cells.desc_set_layout.clone(),
        );
        self.xblur_desc_set = Self::make_xblur_desc_set(
            self.device.clone(),
            self.arena_buffer.clone(),
            self.xblur_buffer.clone(),
            self.xblur.desc_set_layout.clone(),
        );
        self.blit_desc_sets = self
            .swapchain_images
            .iter()
            .cloned()
            .map(|si| {
                Self::make_blit_desc_set(
                    self.device.clone(),
                    self.arena_buffer.clone(),
                    si,
                    self.blit.desc_set_layout.clone(),
                    fmt,
                )
            })
            .collect();
    }

    pub fn randomize_buffer(&mut self, dims: [u32; 2]) {
        let mut trng = rand::thread_rng();
        let data = (std::iter::repeat_with(|| trng.gen::<f32>()))
            .take((dims[0] * dims[1] * 4) as usize)
            .collect::<Vec<_>>();
        let usage = BufferUsage {
            transfer_src: true,
            .. BufferUsage::empty()
        };
        let temp_buf =
            CpuAccessibleBuffer::from_iter(self.device.clone(), usage, false, data.into_iter())
                .expect("could not create CPU-accessible buffer");
        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.compute_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("could not make command buffer builder for buffer randomization!");
        builder
            .copy_buffer(CopyBufferInfoTyped::buffers(
                temp_buf,
                self.arena_buffer.clone(),
            ))
            .expect("could not copy buffer");
        let cmd_buf = builder
            .build()
            .expect("could not build command buffer for randomization!");
        vulkano::sync::now(self.device.clone())
            .then_execute(self.compute_queue.clone(), cmd_buf)
            .expect("could not execute command queue!")
            .then_signal_fence_and_flush()
            .expect("could not flush future!")
            .wait(None)
            .expect("could not wait on future!");
    }

    pub fn do_frame(&mut self, sfc: &Arc<Surface<W>>, dims: [u32; 2]) {
        if let Some(mut last_frame) = self.frame_future.take() {
            last_frame.cleanup_finished();
        }
        let invocation_size = [(dims[0] + 31) / 32, (dims[1] + 31) / 32, 1];
        let buffer_read_params = BufferReadParams {
            height: dims[1] as i32,
            width: dims[0] as i32,
        };
        loop {
            let (image_idx, suboptimal, acquire_future) =
                swapchain::acquire_next_image(self.swapchain.clone(), None)
                    .expect("could not acquire swapchain image!");
            let xblur_cmd_buffer = {
                let mut builder = AutoCommandBufferBuilder::primary(
                    self.device.clone(),
                    self.compute_queue.queue_family_index(),
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
                    .push_constants(
                        self.xblur.pipeline.layout().clone(),
                        0,
                        buffer_read_params.clone(),
                    )
                    .dispatch(invocation_size)
                    .expect("could not dispatch compute operation!");
                builder
                    .build()
                    .expect("could not build blur command buffer!")
            };
            let cells_buffer = {
              let mut builder = AutoCommandBufferBuilder::primary(
                self.device.clone(),
                self.compute_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
              )
                .expect("could not make command buffer builder!");
                builder
                    .bind_pipeline_compute(self.cells.pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        self.cells.pipeline.layout().clone(),
                        0,
                        self.cells_desc_set.clone(),
                    )
                    .push_constants(
                        self.cells.pipeline.layout().clone(),
                        0,
                        buffer_read_params.clone(),
                    )
                    .dispatch(invocation_size)
                    .expect("could not dispatch compute operation!");
                let buffer = builder.build().expect("could not build command buffer!");
                buffer
            };
            let blit_buffer = {
              let mut builder = AutoCommandBufferBuilder::primary(
                self.device.clone(),
                self.compute_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
              )
                .expect("could not make command buffer builder!");
                builder
                    .bind_pipeline_compute(self.blit.pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        self.blit.pipeline.layout().clone(),
                        0,
                        self.blit_desc_sets[image_idx].clone(),
                    )
                    .push_constants(
                        self.blit.pipeline.layout().clone(),
                        0,
                        buffer_read_params.clone(),
                    )
                    .dispatch(invocation_size)
                    .expect("could not dispatch compute operation!");
                let buffer = builder.build().expect("could not build command buffer!");
                buffer
            };
            acquire_future
                .then_execute(self.compute_queue.clone(), xblur_cmd_buffer)
                .expect("could not queue blur operation!")
                .then_signal_fence_and_flush()
                .expect(":(")
                .wait(None)
                .expect(":(");
            // not sure wh it is necessary to force the last operation to
            // complete first. maybe an api bug?
            let present_info = PresentInfo::swapchain(self.swapchain.clone());
            let frame_future = vulkano::sync::now(self.device.clone())
                .then_execute(self.compute_queue.clone(), cells_buffer)
                .expect("could not queue execution of command buffer!")
                .then_execute(self.compute_queue.clone(), blit_buffer)
                .expect("could not queue execution of command buffer!")
                .then_signal_semaphore()
                .then_swapchain_present(
                    self.present_queue.clone(),
                    present_info,
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
