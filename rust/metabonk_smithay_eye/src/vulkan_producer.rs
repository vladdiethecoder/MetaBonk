use std::{
    ffi::CStr,
    os::fd::AsRawFd,
    os::unix::io::RawFd,
};
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use anyhow::Context;
use ash::{
    ext,
    khr,
    vk,
};
use drm_fourcc::{DrmFourcc, DrmModifier};
use nix::ioctl_write_ptr;
use nix::unistd::close;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::{Buffer as _, Fourcc as ShmFourcc};
use smithay::backend::renderer::utils::Buffer as SurfaceBuffer;
use smithay::backend::vulkan::{version::Version, AppInfo, Instance, PhysicalDevice};
use tracing::{debug, info, warn};

// VK_IMAGE_LAYOUT_DRM_FORMAT_MODIFIER_OPTIMAL_EXT isn't exposed as a named constant in our ash version.
// Use the spec value so we can preserve pixels when transitioning imported modifier-backed images.
const IMAGE_LAYOUT_DRM_FORMAT_MODIFIER_OPTIMAL_EXT: vk::ImageLayout = vk::ImageLayout::from_raw(1000158000);

const DMA_BUF_SYNC_READ: u32 = 1 << 0;

fn is_sync_file_signaled(fd: RawFd) -> bool {
    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: pollfd points to a valid struct for the duration of the call.
    let rc = unsafe { libc::poll(&mut pfd as *mut libc::pollfd, 1, 0) };
    if rc <= 0 {
        return false;
    }
    (pfd.revents & libc::POLLIN) != 0
}

fn dmabuf_fdinfo_line(fd: RawFd) -> Option<String> {
    let path = format!("/proc/self/fdinfo/{fd}");
    let txt = std::fs::read_to_string(&path).ok()?;
    let dump_full = std::env::var("METABONK_EYE_FDINFO_FULL")
        .ok()
        .map(|v| v.trim() == "1")
        .unwrap_or(false);
    if dump_full {
        return Some(txt.replace('\n', "\\n"));
    }

    let mut picked: Vec<String> = Vec::new();
    for line in txt.lines() {
        let l = line.trim();
        if l.starts_with("dmabuf:")
            || l.starts_with("dma_buf:")
            || l.starts_with("dma-buf:")
            || l.starts_with("exp_name:")
            || l.starts_with("exp_pid:")
            || l.starts_with("size:")
            || l.starts_with("ino:")
        {
            picked.push(l.to_string());
        }
    }
    if picked.is_empty() {
        None
    } else {
        Some(picked.join(" "))
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct DmaBufImportSyncFile {
    flags: u32,
    fd: i32,
}

ioctl_write_ptr!(dma_buf_import_sync_file, b'b', 3, DmaBufImportSyncFile);

#[derive(Clone, Debug, Default)]
pub struct VkSelect {
    pub device_index: Option<u32>,
    pub device_name_contains: Option<String>,
}

pub struct VulkanProducer {
    instance: Instance,
    physical: PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    cmd_pool: vk::CommandPool,
    ext_drm_mod: ext::image_drm_format_modifier::Device,
    khr_mem_fd: khr::external_memory_fd::Device,
    khr_sema_fd: khr::external_semaphore_fd::Device,
    slots: Vec<Slot>,
    next_slot: usize,
    frame_id: u64,
    width: u32,
    height: u32,
    drm_fourcc: u32,
    vk_format: vk::Format,
    last_src: Option<SrcCache>,
    force_linear_export: bool,
    ignore_consumer_release_wait: bool,
    disable_external_ownership_transfer: bool,
    use_foreign_queue_family: bool,
    debug_dump_staging: bool,
    debug_dump_source: bool,
    debug_dump_after_frame: Option<u64>,
    debug_dump_every_n: Option<u64>,
    debug_dump_max: Option<u64>,
    debug_dump_count: u64,
    debug_dump_done: bool,
}

#[derive(Clone, Debug)]
enum SrcKind {
    Image {
        image: vk::Image,
        memory: vk::DeviceMemory,
        layout: vk::ImageLayout,
    },
    Buffer {
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        row_pitch: u64,
        offset: u64,
    },
}

#[derive(Clone, Debug)]
struct SrcCache {
    key: u64,
    width: u32,
    height: u32,
    kind: SrcKind,
}

#[derive(Clone, Copy, Debug)]
enum ImportedResource {
    Image(vk::Image, vk::DeviceMemory),
    Buffer(vk::Buffer, vk::DeviceMemory),
}

struct Slot {
    image: vk::Image,
    memory: vk::DeviceMemory,
    modifier: u64,
    stride: u32,
    offset: u32,
    mem_size: u64,
    linear_buffer: Option<vk::Buffer>,
    linear_memory: Option<vk::DeviceMemory>,
    linear_size: u64,
    linear_stride: u32,
    cmd: vk::CommandBuffer,
    // Internal semaphore used to chain submissions when we need a second "signal SYNC_FD" submit.
    copy_done: vk::Semaphore,
    fence: vk::Fence,
    prev_acquire: Option<vk::Semaphore>,
    prev_release: Option<vk::Semaphore>,
    to_destroy_sems: Vec<vk::Semaphore>,
    to_destroy: Vec<(vk::Semaphore, vk::Semaphore)>,
    to_destroy_imported: Vec<ImportedResource>,
    to_release_buffers: Vec<SurfaceBuffer>,
}

impl VulkanProducer {
    pub fn new(width: u32, height: u32, format: &str, slots: usize, select: VkSelect) -> anyhow::Result<Self> {
        let app = AppInfo {
            name: "metabonk_smithay_eye".to_string(),
            version: Version::VERSION_1_0,
        };
        let instance = Instance::new(Version::VERSION_1_3, Some(app)).context("create vk instance")?;
        let physical = pick_physical_device(&instance, &select)?;

        let (device, queue_family_index, queue) = create_device(&instance, &physical)?;

        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .context("create command pool")?;

        let ext_drm_mod = ext::image_drm_format_modifier::Device::new(instance.handle(), &device);
        let khr_mem_fd = khr::external_memory_fd::Device::new(instance.handle(), &device);
        let khr_sema_fd = khr::external_semaphore_fd::Device::new(instance.handle(), &device);

        let (drm_fourcc, vk_format) = match format.to_ascii_uppercase().as_str() {
            "ARGB8888" => (DrmFourcc::Argb8888 as u32, vk::Format::B8G8R8A8_UNORM),
            other => anyhow::bail!("unsupported --format {other} (supported: ARGB8888)"),
        };

        let force_linear_export = std::env::var("METABONK_FORCE_LINEAR_EXPORT")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
        if force_linear_export {
            info!("linear export enabled: copying frames into a linear staging buffer");
        }

        let ignore_consumer_release_wait = std::env::var("METABONK_EYE_IGNORE_CONSUMER_RELEASE_WAIT")
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(false);
        if ignore_consumer_release_wait {
            warn!("consumer release waits disabled (METABONK_EYE_IGNORE_CONSUMER_RELEASE_WAIT=1): exported buffers may be overwritten while consumer is reading");
        }

        let disable_external_ownership_transfer = std::env::var("METABONK_EYE_DISABLE_EXTERNAL_OWNERSHIP_TRANSFER")
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(false);
        if disable_external_ownership_transfer {
            warn!(
                "external queue-family ownership transfers disabled (METABONK_EYE_DISABLE_EXTERNAL_OWNERSHIP_TRANSFER=1): relying on semaphore+memory barriers only"
            );
        }

        let mut use_foreign_queue_family = std::env::var("METABONK_EYE_USE_FOREIGN_QUEUE_FAMILY")
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(false);
        if use_foreign_queue_family && !physical.has_device_extension(ext::queue_family_foreign::NAME) {
            warn!(
                "METABONK_EYE_USE_FOREIGN_QUEUE_FAMILY=1 requested but VK_EXT_queue_family_foreign is not supported; falling back to QUEUE_FAMILY_EXTERNAL"
            );
            use_foreign_queue_family = false;
        }
        if use_foreign_queue_family {
            info!("using VK_QUEUE_FAMILY_FOREIGN_EXT for external queue-family ownership transfers");
        }

        let debug_dump_staging = std::env::var("METABONK_DEBUG_DUMP_STAGING")
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(false);
        let debug_dump_source = std::env::var("METABONK_DEBUG_DUMP_SOURCE")
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(false);
        let debug_dump_after_frame = std::env::var("METABONK_DEBUG_DUMP_STAGING_AFTER_FRAME")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|v| *v > 0);
        let debug_dump_every_n = std::env::var("METABONK_DEBUG_DUMP_STAGING_EVERY_N")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|v| *v > 0);
        let debug_dump_max = std::env::var("METABONK_DEBUG_DUMP_STAGING_MAX_DUMPS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|v| *v > 0);
        if debug_dump_staging {
            info!(
                after_frame = debug_dump_after_frame.unwrap_or(1),
                every_n = debug_dump_every_n,
                max_dumps = debug_dump_max,
                "debug staging dump enabled"
            );
        }
        if debug_dump_source {
            info!("debug source dump enabled (dumps occur on the same schedule as staging dumps)");
        }
        let mut producer = Self {
            instance,
            physical,
            device,
            queue,
            queue_family_index,
            cmd_pool,
            ext_drm_mod,
            khr_mem_fd,
            khr_sema_fd,
            slots: Vec::new(),
            next_slot: 0,
            frame_id: 0,
            width,
            height,
            drm_fourcc,
            vk_format,
            last_src: None,
            force_linear_export,
            ignore_consumer_release_wait,
            disable_external_ownership_transfer,
            use_foreign_queue_family,
            debug_dump_staging,
            debug_dump_source,
            debug_dump_after_frame,
            debug_dump_every_n,
            debug_dump_max,
            debug_dump_count: 0,
            debug_dump_done: false,
        };
        producer.init_slots(slots)?;
        Ok(producer)
    }

    pub fn drm_main_device_dev_id(&self) -> Option<libc::dev_t> {
        let node = self
            .physical
            .render_node()
            .ok()
            .flatten()
            .or_else(|| self.physical.primary_node().ok().flatten())?;
        Some(node.dev_id() as libc::dev_t)
    }

    fn init_slots(&mut self, count: usize) -> anyhow::Result<()> {
        let cmds = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(count as u32),
            )
        }
        .context("allocate command buffers")?;

        for idx in 0..count {
            let (image, memory, modifier, stride, offset, mem_size) =
                self.create_exportable_image(self.width, self.height, self.vk_format)?;
            let (linear_buffer, linear_memory, linear_size, linear_stride) = if self.force_linear_export {
                let size = (self.width as u64) * (self.height as u64) * 4;
                let (buf, mem, alloc_size) = self.create_exportable_linear_buffer(size)?;
                (Some(buf), Some(mem), alloc_size, (self.width * 4))
            } else {
                (None, None, 0, 0)
            };
            let copy_done = unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .context("create copy-done semaphore")?;
            let fence = unsafe {
                self.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            }
            .context("create fence")?;
            self.slots.push(Slot {
                image,
                memory,
                modifier,
                stride,
                offset,
                mem_size,
                linear_buffer,
                linear_memory,
                linear_size,
                linear_stride,
                cmd: cmds[idx],
                copy_done,
                fence,
                prev_acquire: None,
                prev_release: None,
                to_destroy_sems: Vec::new(),
                to_destroy: Vec::new(),
                to_destroy_imported: Vec::new(),
                to_release_buffers: Vec::new(),
            });
        }
        Ok(())
    }

    fn create_exportable_image(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> anyhow::Result<(vk::Image, vk::DeviceMemory, u64, u32, u32, u64)> {
        let modifiers = query_modifiers(self.instance.handle(), self.physical.handle(), format)?;
        let mut modifier_list =
            vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(&modifiers);
        // Export to the consumer via OPAQUE_FD (CUDA interop). We still use DRM modifiers for tiling.
        let mut external =
            vk::ExternalMemoryImageCreateInfo::default().handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let mut image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            // Must include TRANSFER_SRC when we later blit/copy out of this image.
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        image_info = image_info.push_next(&mut external).push_next(&mut modifier_list);

        let image = unsafe { self.device.create_image(&image_info, None) }.context("create_image")?;

        // Use the *_requirements2 path to learn whether a dedicated allocation is required for
        // this imported image. Some drivers require MemoryDedicatedAllocateInfo for external
        // memory imports; omitting it can lead to intermittent vkAllocateMemory failures.
        let mut dedicated_req = vk::MemoryDedicatedRequirements::default();
        let mut mem_req2 = vk::MemoryRequirements2::default().push_next(&mut dedicated_req);
        let req_info2 = vk::ImageMemoryRequirementsInfo2::default().image(image);
        unsafe { self.device.get_image_memory_requirements2(&req_info2, &mut mem_req2) };
        let mem_req = mem_req2.memory_requirements;
        let mut export_info =
            vk::ExportMemoryAllocateInfo::default().handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(find_memory_type(self.instance.handle(), self.physical.handle(), mem_req, true)?)
            .push_next(&mut export_info);
        let memory = unsafe { self.device.allocate_memory(&alloc, None) }.context("allocate_memory")?;
        unsafe { self.device.bind_image_memory(image, memory, 0) }.context("bind_image_memory")?;

        // Query actual modifier used.
        let mut mod_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
        unsafe {
            self.ext_drm_mod
                .get_image_drm_format_modifier_properties(image, &mut mod_props)
        }
        .context("get_image_drm_format_modifier_properties")?;
        let modifier = mod_props.drm_format_modifier;

        // For ARGB8888 we expect a single memory plane.
        let sub = vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT);
        let layout = unsafe { self.device.get_image_subresource_layout(image, sub) };
        let offset = layout.offset as u32;
        let stride = layout.row_pitch as u32;
        let mem_size = mem_req.size;

        Ok((image, memory, modifier, stride, offset, mem_size))
    }

    fn create_exportable_linear_buffer(
        &self,
        size: u64,
    ) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory, u64)> {
        let mut external =
            vk::ExternalMemoryBufferCreateInfo::default().handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None) }.context("create_buffer")?;

        let mem_req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mut export_info =
            vk::ExportMemoryAllocateInfo::default().handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(find_memory_type(self.instance.handle(), self.physical.handle(), mem_req, true)?)
            .push_next(&mut export_info);
        let memory = unsafe { self.device.allocate_memory(&alloc, None) }.context("allocate_memory(buffer)")?;
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }.context("bind_buffer_memory")?;

        Ok((buffer, memory, mem_req.size))
    }

    pub fn render_next(&mut self, clear_phase: u8) -> anyhow::Result<RenderedFrame> {
        let slot_idx = self.next_slot % self.slots.len();
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        {
            let fence = self.slots[slot_idx].fence;
            // Wait for previous GPU work on this slot (allows safe semaphore destruction/reset).
            self.wait_for_fence(fence, "Render", slot_idx).context("wait_for_fences")?;
            let slot = &mut self.slots[slot_idx];
            unsafe { self.device.reset_fences(&[slot.fence]) }.ok();
            for sem in slot.to_destroy_sems.drain(..) {
                unsafe {
                    self.device.destroy_semaphore(sem, None);
                }
            }
            for (a, r) in slot.to_destroy.drain(..) {
                unsafe {
                    self.device.destroy_semaphore(a, None);
                    self.device.destroy_semaphore(r, None);
                }
            }
            for res in slot.to_destroy_imported.drain(..) {
                unsafe {
                    match res {
                        ImportedResource::Image(img, mem) => {
                            self.device.destroy_image(img, None);
                            self.device.free_memory(mem, None);
                        }
                        ImportedResource::Buffer(buf, mem) => {
                            self.device.destroy_buffer(buf, None);
                            self.device.free_memory(mem, None);
                        }
                    }
                }
            }
            // Drop any held Wayland buffers now that GPU work on this slot completed.
            slot.to_release_buffers.clear();
        }

        let (acquire, acquire_fd) = self.create_exportable_semaphore()?;
        let (release, release_fd) = self.create_exportable_semaphore()?;

        // Record: clear image to a varying color (keeps MVP visually obvious for debugging).
        let (
            image,
            cmd,
            fence,
            prev_acquire,
            prev_release,
            modifier,
            stride,
            offset,
            mem_size,
            memory,
            linear_buffer,
            linear_memory,
            linear_size,
            linear_stride,
        ) = {
            let slot = &mut self.slots[slot_idx];
            (
                slot.image,
                slot.cmd,
                slot.fence,
                slot.prev_acquire,
                slot.prev_release,
                slot.modifier,
                slot.stride,
                slot.offset,
                slot.mem_size,
                slot.memory,
                slot.linear_buffer,
                slot.linear_memory,
                slot.linear_size,
                slot.linear_stride,
            )
        };
        let use_linear_export = self.force_linear_export && linear_buffer.is_some() && linear_memory.is_some();

        unsafe { self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }.ok();
        unsafe {
            self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let to_transfer = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image)
                .subresource_range(range);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer],
            );
            let c = clear_color(clear_phase);
            self.device
                .cmd_clear_color_image(cmd, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &c, &[range]);
            if use_linear_export {
                let to_src = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(image)
                    .subresource_range(range);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_src],
                );

                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1))
                    .image_extent(vk::Extent3D {
                        width: self.width,
                        height: self.height,
                        depth: 1,
                    });
                if let Some(buf) = linear_buffer {
                    self.device.cmd_copy_image_to_buffer(
                        cmd,
                        image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        buf,
                        &[region],
                    );
                }

                let to_general = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(image)
                    .subresource_range(range);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
            } else {
                let to_general = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(image)
                    .subresource_range(range);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
            }
            self.device.end_command_buffer(cmd)?;
        }

        // Submit with explicit wait on previous consumer release.
        let mut wait_sems: Vec<vk::Semaphore> = Vec::new();
        let mut wait_stages: Vec<vk::PipelineStageFlags> = Vec::new();
        if let Some(prev_rel) = prev_release {
            if !self.ignore_consumer_release_wait {
                wait_sems.push(prev_rel);
                wait_stages.push(vk::PipelineStageFlags::TRANSFER);
            }
        }
        let signal_sems = [acquire];

        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&cmd))
            .signal_semaphores(&signal_sems);

        unsafe { self.device.queue_submit(self.queue, &[submit], fence) }.context("queue_submit")?;

        // Export DMABuf FD for the memory backing the exported surface.
        let (out_memory, out_modifier, out_stride, out_offset, out_size) = if use_linear_export {
            (
                linear_memory.expect("linear export memory missing"),
                u64::from(DrmModifier::Linear),
                linear_stride,
                0,
                linear_size,
            )
        } else {
            (memory, modifier, stride, offset, mem_size)
        };
        let dmabuf_fd = self.export_memory_dmabuf_fd(out_memory)?;

        debug!(
            "export frame: modifier=0x{:016x} fourcc=0x{:08x} stride={} offset={} size_bytes={}",
            modifier,
            self.drm_fourcc,
            stride,
            offset,
            mem_size
        );

        debug!(
            "export frame: modifier=0x{:016x} fourcc=0x{:08x} stride={} offset={} size_bytes={}",
            modifier,
            self.drm_fourcc,
            stride,
            offset,
            mem_size
        );

        self.frame_id += 1;
        {
            let slot = &mut self.slots[slot_idx];
            if let (Some(pa), Some(pr)) = (prev_acquire, prev_release) {
                slot.to_destroy.push((pa, pr));
            }
            slot.prev_acquire = Some(acquire);
            slot.prev_release = Some(release);
        }

        Ok(RenderedFrame {
            frame_id: self.frame_id,
            width: self.width,
            height: self.height,
            drm_fourcc: self.drm_fourcc,
            modifier: out_modifier,
            stride: out_stride,
            offset: out_offset,
            size_bytes: out_size as u32,
            dmabuf_fd,
            acquire_fence_fd: acquire_fd,
            release_fence_fd: release_fd,
        })
    }

    pub fn render_from_dmabuf(
        &mut self,
        src: &Dmabuf,
        wait_fence_fd: Option<RawFd>,
        buffer_hold: Option<SurfaceBuffer>,
    ) -> anyhow::Result<RenderedFrame> {
        struct FdGuard(Option<RawFd>);
        impl Drop for FdGuard {
            fn drop(&mut self) {
                if let Some(fd) = self.0.take() {
                    let _ = close(fd);
                }
            }
        }

        info!(wait_fence_fd = ?wait_fence_fd, "render_from_dmabuf acquire sync fd");
        let mut wait_fd = FdGuard(wait_fence_fd);
        let src_dmabuf_fd = src.handles().next().map(|h| h.as_raw_fd()).unwrap_or(-1);
        // When the producer relies on implicit sync (no explicit release fences), we must attach a
        // READ fence back onto the source dma-buf reservation object; otherwise the producer can
        // begin writing the next frame into the same buffer while we're still copying it, which
        // manifests as horizontal tearing/striping.
        //
        // Default to enabled for correctness; allow opting out for debugging via env var.
        let import_read_fence = std::env::var("METABONK_EYE_IMPORT_READ_FENCE")
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(true);
        let need_src_read_fence = import_read_fence && src_dmabuf_fd >= 0;
        let slot_idx = self.next_slot % self.slots.len();
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        {
            let fence = self.slots[slot_idx].fence;
            self.wait_for_fence(fence, "Blit", slot_idx).context("wait_for_fences")?;
            let slot = &mut self.slots[slot_idx];
            unsafe { self.device.reset_fences(&[slot.fence]) }.ok();
            for sem in slot.to_destroy_sems.drain(..) {
                unsafe {
                    self.device.destroy_semaphore(sem, None);
                }
            }
            for (a, r) in slot.to_destroy.drain(..) {
                unsafe {
                    self.device.destroy_semaphore(a, None);
                    self.device.destroy_semaphore(r, None);
                }
            }
            for res in slot.to_destroy_imported.drain(..) {
                unsafe {
                    match res {
                        ImportedResource::Image(img, mem) => {
                            self.device.destroy_image(img, None);
                            self.device.free_memory(mem, None);
                        }
                        ImportedResource::Buffer(buf, mem) => {
                            self.device.destroy_buffer(buf, None);
                            self.device.free_memory(mem, None);
                        }
                    }
                }
            }
            // Drop any held Wayland buffers now that GPU work on this slot completed.
            slot.to_release_buffers.clear();
        }

        let (acquire, acquire_fd) = self.create_exportable_semaphore()?;
        let (release, release_fd) = self.create_exportable_semaphore()?;

        // Import the source DMA-BUF as a Vulkan image for GPU-to-GPU copy into our exportable slot image.
        //
        // Critical: do NOT import/allocate a fresh Vulkan image for every frame. Even if the producer
        // is reusing a small set of DMA-BUFs (triple buffering), importing every tick can exhaust
        // driver resources and lead to ERROR_OUT_OF_DEVICE_MEMORY. Reuse the last import until the
        // DMA-BUF identity changes.
        let modifier_u64: u64 = src.format().modifier.into();
        let modifier_is_linear = modifier_u64 == 0;
        let prefer_buffer_import = std::env::var("METABONK_EYE_PREFER_BUFFER_IMPORT")
            .ok()
            .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));
        let src_key = dmabuf_key(src);
        let mut imported_new = false;
        let (src_kind, src_w, src_h) = if let Some(c) = self.last_src.as_ref() {
            if c.key == src_key {
                (c.kind.clone(), c.width, c.height)
            } else {
                // Defer destruction of the previous import until the destination fence signals.
                if let Some(old) = self.last_src.take() {
                    let slot = &mut self.slots[slot_idx];
                    match old.kind {
                        SrcKind::Image { image, memory, .. } => {
                            slot.to_destroy_imported.push(ImportedResource::Image(image, memory));
                        }
                        SrcKind::Buffer { buffer, memory, .. } => {
                            slot.to_destroy_imported.push(ImportedResource::Buffer(buffer, memory));
                        }
                    }
                }
                let (kind, w, h) = if modifier_is_linear && prefer_buffer_import {
                    match self.import_dmabuf_buffer(src) {
                        Ok((buf, mem, w, h, row_pitch, offset)) => {
                            (SrcKind::Buffer { buffer: buf, memory: mem, row_pitch, offset }, w, h)
                        }
                        Err(e) => {
                            warn!("linear dmabuf buffer import failed; falling back to image import: {e}");
                            let (img, mem, w, h, layout) = self.import_dmabuf_image(src)?;
                            (
                                SrcKind::Image {
                                    image: img,
                                    memory: mem,
                                    // Imported dma-bufs originate outside Vulkan; treat them as
                                    // already containing valid pixels in their modifier-optimal
                                    // layout (do NOT use UNDEFINED, which permits content discard).
                                    layout,
                                },
                                w,
                                h,
                            )
                        }
                    }
                } else {
                    // Prefer image import for correctness; buffer import is an optimization and may
                    // misinterpret tiling/modifier metadata on some driver stacks.
                    match self.import_dmabuf_image(src) {
                        Ok((img, mem, w, h, layout)) => (
                            SrcKind::Image {
                                image: img,
                                memory: mem,
                                // Imported dma-bufs originate outside Vulkan; treat them as already
                                // containing valid pixels in their modifier-optimal layout.
                                layout,
                            },
                            w,
                            h,
                        ),
                        Err(e_img) if modifier_is_linear => {
                            warn!("linear dmabuf image import failed; falling back to buffer import: {e_img}");
                            let (buf, mem, w, h, row_pitch, offset) = self.import_dmabuf_buffer(src)?;
                            (SrcKind::Buffer { buffer: buf, memory: mem, row_pitch, offset }, w, h)
                        }
                        Err(e_img) => return Err(e_img),
                    }
                };
                imported_new = true;
                self.last_src = Some(SrcCache {
                    key: src_key,
                    width: w,
                    height: h,
                    kind: kind.clone(),
                });
                (kind, w, h)
            }
        } else {
            let (kind, w, h) = if modifier_is_linear && prefer_buffer_import {
                match self.import_dmabuf_buffer(src) {
                    Ok((buf, mem, w, h, row_pitch, offset)) => {
                        (SrcKind::Buffer { buffer: buf, memory: mem, row_pitch, offset }, w, h)
                    }
                    Err(e) => {
                        warn!("linear dmabuf buffer import failed; falling back to image import: {e}");
                        let (img, mem, w, h, layout) = self.import_dmabuf_image(src)?;
                        (
                            SrcKind::Image {
                                image: img,
                                memory: mem,
                                // Imported dma-bufs originate outside Vulkan; treat them as already
                                // containing valid pixels in their modifier-optimal layout.
                                layout,
                            },
                            w,
                            h,
                        )
                    }
                }
            } else {
                match self.import_dmabuf_image(src) {
                    Ok((img, mem, w, h, layout)) => (
                        SrcKind::Image {
                            image: img,
                            memory: mem,
                            // Imported dma-bufs originate outside Vulkan; treat them as already
                            // containing valid pixels in their modifier-optimal layout.
                            layout,
                        },
                        w,
                        h,
                    ),
                    Err(e_img) if modifier_is_linear => {
                        warn!("linear dmabuf image import failed; falling back to buffer import: {e_img}");
                        let (buf, mem, w, h, row_pitch, offset) = self.import_dmabuf_buffer(src)?;
                        (SrcKind::Buffer { buffer: buf, memory: mem, row_pitch, offset }, w, h)
                    }
                    Err(e_img) => return Err(e_img),
                }
            };
            imported_new = true;
            self.last_src = Some(SrcCache {
                key: src_key,
                width: w,
                height: h,
                kind: kind.clone(),
            });
            (kind, w, h)
        };

        if imported_new {
            let fmt = src.format().code;
            let plane_count = src.num_planes();
            let stride0 = src.strides().next().unwrap_or(0);
            let offset0 = src.offsets().next().unwrap_or(0);
            match &src_kind {
                SrcKind::Buffer {
                    row_pitch, offset, ..
                } => {
                    info!(
                        ?fmt,
                        modifier = format_args!("0x{modifier_u64:016x}"),
                        plane_count,
                        stride0,
                        offset0,
                        row_pitch,
                        buffer_offset = offset,
                        "dmabuf import: using BUFFER path"
                    );
                }
                SrcKind::Image { layout, .. } => {
                    info!(
                        ?fmt,
                        modifier = format_args!("0x{modifier_u64:016x}"),
                        plane_count,
                        stride0,
                        offset0,
                        ?layout,
                        "dmabuf import: using IMAGE path"
                    );
                }
            }

            for (idx, handle) in src.handles().enumerate() {
                let fd = handle.as_raw_fd();
                if let Some(line) = dmabuf_fdinfo_line(fd) {
                    info!(plane = idx, fd, fdinfo = %line, "dmabuf plane fdinfo");
                } else {
                    debug!(plane = idx, fd, "dmabuf plane fdinfo missing");
                }
            }
        }

        let (
            dst_image,
            cmd,
            fence,
            prev_acquire,
            prev_release,
            modifier,
            stride,
            offset,
            mem_size,
            memory,
            linear_buffer,
            linear_memory,
            linear_size,
            linear_stride,
        ) = {
            let slot = &mut self.slots[slot_idx];
            (
                slot.image,
                slot.cmd,
                slot.fence,
                slot.prev_acquire,
                slot.prev_release,
                slot.modifier,
                slot.stride,
                slot.offset,
                slot.mem_size,
                slot.memory,
                slot.linear_buffer,
                slot.linear_memory,
                slot.linear_size,
                slot.linear_stride,
            )
        };
        let use_linear_export = self.force_linear_export && linear_buffer.is_some() && linear_memory.is_some();
        let frame_id = self.frame_id + 1;
        let debug_dump_enabled = self.should_debug_dump_staging(frame_id, use_linear_export);
        let debug_dump_source_enabled = debug_dump_enabled && self.debug_dump_source;
        let mut debug_buffer: Option<vk::Buffer> = None;
        let mut debug_memory: Option<vk::DeviceMemory> = None;
        let mut debug_src_buffer: Option<vk::Buffer> = None;
        let mut debug_src_memory: Option<vk::DeviceMemory> = None;
        let mut debug_src_size: Option<u64> = None;
        let mut debug_src_row_bytes: Option<usize> = None;
        let debug_buffer_size = (u64::from(self.width)) * (u64::from(self.height)) * 4;
        if debug_dump_enabled {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(debug_buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = unsafe { self.device.create_buffer(&buffer_info, None) }
                .context("create debug staging buffer")?;
            let mem_req = unsafe { self.device.get_buffer_memory_requirements(buf) };
            let mem_type = find_memory_type_with_flags(
                self.instance.handle(),
                self.physical.handle(),
                mem_req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .context("find debug staging buffer memory type")?;
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_req.size)
                .memory_type_index(mem_type);
            let mem = unsafe { self.device.allocate_memory(&alloc, None) }
                .context("allocate debug staging buffer memory")?;
            unsafe { self.device.bind_buffer_memory(buf, mem, 0) }
                .context("bind debug staging buffer memory")?;
            debug_buffer = Some(buf);
            debug_memory = Some(mem);
        }
        if debug_dump_source_enabled {
            let (size_bytes, row_bytes) = match &src_kind {
                SrcKind::Image { .. } => {
                    let row_bytes = (self.width as usize).saturating_mul(4);
                    let size_bytes = (row_bytes as u64).saturating_mul(self.height as u64);
                    (size_bytes, row_bytes)
                }
                SrcKind::Buffer { row_pitch, .. } => {
                    let row_bytes = (*row_pitch).try_into().unwrap_or(0usize);
                    let size_bytes = row_pitch.saturating_mul(self.height as u64);
                    (size_bytes, row_bytes)
                }
            };
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size_bytes.max(debug_buffer_size))
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = unsafe { self.device.create_buffer(&buffer_info, None) }
                .context("create debug source buffer")?;
            let mem_req = unsafe { self.device.get_buffer_memory_requirements(buf) };
            let mem_type = find_memory_type_with_flags(
                self.instance.handle(),
                self.physical.handle(),
                mem_req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .context("find debug source buffer memory type")?;
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_req.size)
                .memory_type_index(mem_type);
            let mem = unsafe { self.device.allocate_memory(&alloc, None) }
                .context("allocate debug source buffer memory")?;
            unsafe { self.device.bind_buffer_memory(buf, mem, 0) }
                .context("bind debug source buffer memory")?;
            debug_src_buffer = Some(buf);
            debug_src_memory = Some(mem);
            debug_src_size = Some(size_bytes);
            debug_src_row_bytes = Some(row_bytes);
        }

        if src_w != self.width || src_h != self.height {
            // For now, require a matching input size. (This avoids an extra blit path and keeps the ABI stable.)
            // We still send a RESET so the worker drops the segment rather than training on a frozen eye.
            if imported_new {
                // Not used by any GPU submission yet; safe to destroy immediately.
                unsafe {
                    match src_kind {
                        SrcKind::Image { image, memory, .. } => {
                            self.device.destroy_image(image, None);
                            self.device.free_memory(memory, None);
                        }
                        SrcKind::Buffer { buffer, memory, .. } => {
                            self.device.destroy_buffer(buffer, None);
                            self.device.free_memory(memory, None);
                        }
                    }
                }
                self.last_src = None;
            }
            anyhow::bail!(
                "source dmabuf size {}x{} does not match eye size {}x{}",
                src_w,
                src_h,
                self.width,
                self.height
            );
        }

        unsafe { self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }.ok();
        unsafe {
            self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;

            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);

            match src_kind {
                SrcKind::Image {
                    image: src_image,
                    layout: src_layout,
                    ..
                } => {
                    // Some driver stacks appear to mishandle DMA-BUF images when transitioning into
                    // TRANSFER_SRC_OPTIMAL (producing stable "banding"). Use GENERAL for transfer
                    // operations to keep the imported modifier-backed image in a more conservative
                    // layout that preserves contents across the transition.
                    let src_copy_layout = vk::ImageLayout::GENERAL;
                    let src_release_layout = if src_layout == vk::ImageLayout::PREINITIALIZED {
                        // PREINITIALIZED is only valid as an initial layout; once transitioned away
                        // it must not be used again. Keep the imported image in GENERAL between
                        // frames after the first transition.
                        vk::ImageLayout::GENERAL
                    } else {
                        src_layout
                    };
                    let foreign_qfi = if self.use_foreign_queue_family {
                        vk::QUEUE_FAMILY_FOREIGN_EXT
                    } else {
                        vk::QUEUE_FAMILY_EXTERNAL
                    };
                    let (src_qfi, dst_qfi, rel_src_qfi, rel_dst_qfi) = if self.disable_external_ownership_transfer {
                        (
                            vk::QUEUE_FAMILY_IGNORED,
                            vk::QUEUE_FAMILY_IGNORED,
                            vk::QUEUE_FAMILY_IGNORED,
                            vk::QUEUE_FAMILY_IGNORED,
                        )
                    } else {
                        (
                            foreign_qfi,
                            self.queue_family_index,
                            self.queue_family_index,
                            foreign_qfi,
                        )
                    };
                    let src_to_src = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .old_layout(src_layout)
                        .new_layout(src_copy_layout)
                        // Imported DMA-BUF images originate outside Vulkan. Perform an explicit
                        // queue-family ownership transfer from EXTERNAL -> our queue family to
                        // make external writes visible and avoid undefined reads (banding/tearing).
                        .src_queue_family_index(src_qfi)
                        .dst_queue_family_index(dst_qfi)
                        .image(src_image)
                        .subresource_range(range);

                    let dst_to_dst = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::MEMORY_READ)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .image(dst_image)
                        .subresource_range(range);

                    self.device.cmd_pipeline_barrier(
                        cmd,
                        // Acquire ownership of the external image at the top of the pipeline.
                        // External producers can write at any stage; using TOP_OF_PIPE here avoids
                        // under-scoping the dependency and has been observed to fix stable
                        // "banding" corruption on some driver stacks.
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[src_to_src, dst_to_dst],
                    );

                    let sub = vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1);
                    let region = vk::ImageCopy::default()
                        .src_subresource(sub)
                        .dst_subresource(sub)
                        .extent(vk::Extent3D {
                            width: self.width,
                            height: self.height,
                            depth: 1,
                        });
                    if let Some(buf) = debug_src_buffer {
                        let src_region = vk::BufferImageCopy::default()
                            .buffer_offset(0)
                            .buffer_row_length(0)
                            .buffer_image_height(0)
                            .image_subresource(sub)
                            .image_extent(vk::Extent3D {
                                width: self.width,
                                height: self.height,
                                depth: 1,
                            });
                        self.device.cmd_copy_image_to_buffer(
                            cmd,
                            src_image,
                            src_copy_layout,
                            buf,
                            &[src_region],
                        );
                    }
                    self.device.cmd_copy_image(
                        cmd,
                        src_image,
                        src_copy_layout,
                        dst_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                    );

                    let dst_to_general = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(dst_image)
                        .subresource_range(range);

                    let src_back = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .dst_access_mask(vk::AccessFlags::empty())
                        .old_layout(src_copy_layout)
                        .new_layout(src_release_layout)
                        // Release ownership back to the external producer.
                        .src_queue_family_index(rel_src_qfi)
                        .dst_queue_family_index(rel_dst_qfi)
                        .image(src_image)
                        .subresource_range(range);

                    self.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[src_back, dst_to_general],
                    );
                }
                SrcKind::Buffer { buffer, row_pitch, offset, .. } => {
                    let dst_to_dst = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::MEMORY_READ)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .image(dst_image)
                        .subresource_range(range);

                    self.device.cmd_pipeline_barrier(
                        cmd,
                        // See image path: acquire dependency at TOP_OF_PIPE to avoid under-scoping
                        // external producer writes.
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[dst_to_dst],
                    );

                    let sub = vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1);
                    if let Some(buf) = debug_src_buffer {
                        // Dump raw source bytes (including padding) starting at the plane offset.
                        let size_bytes = debug_src_size.unwrap_or(debug_buffer_size);
                        let copy = vk::BufferCopy::default()
                            .src_offset(offset)
                            .dst_offset(0)
                            .size(size_bytes);
                        self.device.cmd_copy_buffer(cmd, buffer, buf, &[copy]);
                    }
                    let row_length = (row_pitch / 4).max(self.width as u64) as u32;
                    let region = vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .buffer_row_length(row_length)
                        .buffer_image_height(self.height)
                        .image_subresource(sub)
                        .image_extent(vk::Extent3D {
                            width: self.width,
                            height: self.height,
                            depth: 1,
                        });
                    self.device.cmd_copy_buffer_to_image(
                        cmd,
                        buffer,
                        dst_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                    );

                    let dst_to_general = vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(dst_image)
                        .subresource_range(range);

                    self.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[dst_to_general],
                    );
                }
            }

            if use_linear_export {
                let to_src = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_READ)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(dst_image)
                    .subresource_range(range);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_src],
                );

                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: self.width,
                        height: self.height,
                        depth: 1,
                    });
                if let Some(buf) = linear_buffer {
                    self.device.cmd_copy_image_to_buffer(
                        cmd,
                        dst_image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        buf,
                        &[region],
                    );
                }
                if let Some(buf) = debug_buffer {
                    self.device.cmd_copy_image_to_buffer(
                        cmd,
                        dst_image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        buf,
                        &[region],
                    );
                }
                let to_general = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(dst_image)
                    .subresource_range(range);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_general],
                );
            }

            self.device.end_command_buffer(cmd)?;
        }
        // Update cached source layout after the first use. PREINITIALIZED is only valid as an
        // initial layout; once transitioned away it must not be used again.
        if let Some(c) = self.last_src.as_mut() {
            if c.key == src_key {
                if let SrcKind::Image { layout, .. } = &mut c.kind {
                    if *layout == vk::ImageLayout::PREINITIALIZED {
                        *layout = vk::ImageLayout::GENERAL;
                    }
                }
            }
        }

        let mut imported_wait_sem: Option<vk::Semaphore> = None;
        if let Some(fd) = wait_fd.0.take() {
            info!(
                fd,
                signaled = is_sync_file_signaled(fd),
                "importing acquire sync fd into semaphore"
            );
            match self.import_sync_fd_semaphore(fd) {
                Ok(sem) => {
                    info!(fd, "acquire sync fd import succeeded");
                    imported_wait_sem = Some(sem);
                    // Vulkan owns the fd after a successful import.
                }
                Err(err) => {
                    warn!(fd, "failed to import acquire sync fd into semaphore: {err}");
                    let _ = close(fd);
                }
            }
        } else {
            info!("no acquire sync fd provided for render_from_dmabuf");
        }

        // Submit with explicit wait on previous consumer release.
        let mut wait_sems: Vec<vk::Semaphore> = Vec::new();
        let mut wait_stages: Vec<vk::PipelineStageFlags> = Vec::new();
        if let Some(prev_rel) = prev_release {
            if !self.ignore_consumer_release_wait {
                wait_sems.push(prev_rel);
                // Previous consumer operations may touch the image in any stage. Waiting at
                // TOP_OF_PIPE prevents us from starting acquisition/layout transitions early.
                wait_stages.push(vk::PipelineStageFlags::TOP_OF_PIPE);
            }
        }
        if let Some(wait_sem) = imported_wait_sem {
            wait_sems.push(wait_sem);
            // Imported sync_file fences represent external GPU work. Wait at TOP_OF_PIPE so the
            // subsequent queue-family ownership transfer and layout transitions are also ordered
            // after the fence signals.
            wait_stages.push(vk::PipelineStageFlags::TOP_OF_PIPE);
        }
        // Optional implicit-sync bridge (step 3 of dma-buf explicit interop):
        // attach our GPU READ completion fence to the *source* dma-buf reservation object, so the
        // producer won't start writing the next frame into this buffer while we're still copying.
        //
        // Important: do not signal a SYNC_FD semaphore in the same submission that signals the
        // OPAQUE_FD acquire semaphore (driver stacks have been observed to wedge in that case).
        //
        // We instead do:
        // - Submit #1: wait on producer fence, copy src->dst, signal `src_readback_sem` (SYNC_FD)
        //             and `copy_done` (internal).
        // - Immediately export `src_readback_sem` as a sync_file and IMPORT_SYNC_FILE it back into
        //   the source dmabuf as a READ fence (while the copy is still in-flight).
        // - Submit #2: wait on `copy_done`, signal `acquire` (OPAQUE_FD) for the consumer, and
        //             signal the slot fence.
        let src_readback_sem = if need_src_read_fence {
            Some(self.create_sync_fd_export_semaphore()?)
        } else {
            None
        };
        let copy_done = self.slots[slot_idx].copy_done;
        let mut signal_sems: Vec<vk::Semaphore> = Vec::new();
        if let Some(sem) = src_readback_sem {
            signal_sems.push(sem);
            signal_sems.push(copy_done);
        } else {
            signal_sems.push(acquire);
        }

        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&cmd))
            .signal_semaphores(&signal_sems);

        // If we need a second submit (to isolate SYNC_FD from OPAQUE_FD), use a NULL fence here and
        // attach the slot fence to the second submit (so slot recycling waits for both submissions).
        let fence1 = if src_readback_sem.is_some() { vk::Fence::null() } else { fence };
        unsafe { self.device.queue_submit(self.queue, &[submit], fence1) }.context("queue_submit")?;

        if let Some(buf) = buffer_hold {
            let slot = &mut self.slots[slot_idx];
            slot.to_release_buffers.push(buf);
        }

        if let Some(sem) = src_readback_sem {
            // Export a sync_file for the in-flight read completion and attach it to the source
            // dmabuf reservation object. This prevents the producer from overwriting while our
            // copy is still executing.
            match self.export_semaphore_sync_fd(sem) {
                Ok(fd) if fd >= 0 => {
                    info!(src_dmabuf_fd, fd, "exported source readback sync_file fd; importing into dmabuf as READ fence");
                    if let Err(e) = self.import_dmabuf_read_fence(src_dmabuf_fd, fd) {
                        warn!(src_dmabuf_fd, fd, "failed to import read fence into source dmabuf: {e}");
                    }
                    let _ = close(fd);
                }
                Ok(fd) => {
                    info!(fd, "exported source readback sync_file fd already signaled; skipping dmabuf import");
                }
                Err(e) => {
                    warn!("failed to export sync fd from source readback semaphore: {e}");
                }
            }

            // Second submit: wait for the main copy to complete (copy_done), then signal `acquire`
            // (OPAQUE_FD) for the consumer. This isolates handle types across submissions.
            let wait_sems2 = [copy_done];
            let wait_stages2 = [vk::PipelineStageFlags::TRANSFER];
            let signal_sems2 = [acquire];
            let submit2 = vk::SubmitInfo::default()
                .wait_semaphores(&wait_sems2)
                .wait_dst_stage_mask(&wait_stages2)
                .command_buffers(&[])
                .signal_semaphores(&signal_sems2);
            unsafe { self.device.queue_submit(self.queue, &[submit2], fence) }.context("queue_submit(acquire)")?;

            let slot = &mut self.slots[slot_idx];
            slot.to_destroy_sems.push(sem);
        }

        if let Some(wait_sem) = imported_wait_sem {
            let slot = &mut self.slots[slot_idx];
            slot.to_destroy_sems.push(wait_sem);
        }

        if debug_dump_enabled {
            // Ensure GPU completed the blit/copy before mapping the debug buffer.
            if let Err(e) = self.wait_for_fence(fence, "DebugDump", slot_idx) {
                warn!("debug staging dump skipped (fence wait failed): {e}");
            } else {
                if let (Some(buf), Some(mem), Some(size), Some(row_bytes)) =
                    (debug_src_buffer, debug_src_memory, debug_src_size, debug_src_row_bytes)
                {
                    if let Err(e) = self.debug_dump_buffer_ppm_kind(mem, size, row_bytes, frame_id, "source") {
                        warn!("debug source dump failed: {e}");
                    }
                    unsafe {
                        self.device.destroy_buffer(buf, None);
                        self.device.free_memory(mem, None);
                    }
                }
                if let (Some(buf), Some(mem)) = (debug_buffer, debug_memory) {
                    if let Err(e) = self.debug_dump_buffer_ppm(mem, debug_buffer_size, frame_id) {
                        warn!("debug staging dump failed: {e}");
                    }
                    unsafe {
                        self.device.destroy_buffer(buf, None);
                        self.device.free_memory(mem, None);
                    }
                    self.debug_dump_count = self.debug_dump_count.saturating_add(1);
                    if self.debug_dump_every_n.is_none() {
                        self.debug_dump_done = true;
                    } else if let Some(max) = self.debug_dump_max {
                        if self.debug_dump_count >= max {
                            info!(
                                max_dumps = max,
                                "debug staging dump limit reached; disabling further dumps"
                            );
                            self.debug_dump_done = true;
                        }
                    }
                }
            }
        }

        // Export DMABuf FD for the memory backing the exported surface.
        let (out_memory, out_modifier, out_stride, out_offset, out_size) = if use_linear_export {
            (
                linear_memory.expect("linear export memory missing"),
                u64::from(DrmModifier::Linear),
                linear_stride,
                0,
                linear_size,
            )
        } else {
            (memory, modifier, stride, offset, mem_size)
        };
        let dmabuf_fd = self.export_memory_dmabuf_fd(out_memory)?;

        self.frame_id += 1;
        {
            let slot = &mut self.slots[slot_idx];
            if let (Some(pa), Some(pr)) = (prev_acquire, prev_release) {
                slot.to_destroy.push((pa, pr));
            }
            slot.prev_acquire = Some(acquire);
            slot.prev_release = Some(release);
        }

        Ok(RenderedFrame {
            frame_id: self.frame_id,
            width: self.width,
            height: self.height,
            drm_fourcc: self.drm_fourcc,
            modifier: out_modifier,
            stride: out_stride,
            offset: out_offset,
            size_bytes: out_size as u32,
            dmabuf_fd,
            acquire_fence_fd: acquire_fd,
            release_fence_fd: release_fd,
        })
    }

    /// GPU-only passthrough mode: forward the source DMA-BUF FD to the worker without importing it into Vulkan.
    ///
    /// This avoids driver-specific DMA-BUF → Vulkan import failures while preserving the DMA-BUF+fence ABI.
    /// The acquire fence is signaled by a trivial Vulkan submission (no-op), and the producer waits on the
    /// previous consumer release fence to provide backpressure.
    pub fn render_passthrough_dmabuf(
        &mut self,
        src: &Dmabuf,
        override_size: Option<(u32, u32)>,
        wait_fence_fd: Option<RawFd>,
        buffer_hold: Option<SurfaceBuffer>,
    ) -> anyhow::Result<RenderedFrame> {
        struct FdGuard(Option<RawFd>);
        impl Drop for FdGuard {
            fn drop(&mut self) {
                if let Some(fd) = self.0.take() {
                    let _ = close(fd);
                }
            }
        }

        let mut wait_fd = FdGuard(wait_fence_fd);
        if src.num_planes() != 1 {
            anyhow::bail!("only single-plane dmabufs are supported (got {} planes)", src.num_planes());
        }

        let slot_idx = self.next_slot % self.slots.len();
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        {
            let fence = self.slots[slot_idx].fence;
            self.wait_for_fence(fence, "Passthrough", slot_idx).context("wait_for_fences")?;
            let slot = &mut self.slots[slot_idx];
            unsafe { self.device.reset_fences(&[slot.fence]) }.ok();
            for sem in slot.to_destroy_sems.drain(..) {
                unsafe {
                    self.device.destroy_semaphore(sem, None);
                }
            }
            for (a, r) in slot.to_destroy.drain(..) {
                unsafe {
                    self.device.destroy_semaphore(a, None);
                    self.device.destroy_semaphore(r, None);
                }
            }
            for res in slot.to_destroy_imported.drain(..) {
                unsafe {
                    match res {
                        ImportedResource::Image(img, mem) => {
                            self.device.destroy_image(img, None);
                            self.device.free_memory(mem, None);
                        }
                        ImportedResource::Buffer(buf, mem) => {
                            self.device.destroy_buffer(buf, None);
                            self.device.free_memory(mem, None);
                        }
                    }
                }
            }
            slot.to_release_buffers.clear();
        }

        let src_fd = src.handles().next().unwrap();
        let dmabuf_fd = unsafe { libc::dup(src_fd.as_raw_fd()) };
        if dmabuf_fd < 0 {
            return Err(std::io::Error::last_os_error()).context("dup source dmabuf fd");
        }

        let size = src.size();
        let mut width = size.w as u32;
        let mut height = size.h as u32;
        if (width == 0 || height == 0) && override_size.is_some() {
            let (ow, oh) = override_size.unwrap();
            width = ow;
            height = oh;
        }
        let modifier: u64 = src.format().modifier.into();
        let offset = src.offsets().next().unwrap_or(0) as u32;
        let mut stride = src.strides().next().unwrap_or(0) as u32;
        // Some producers (notably certain XWayland paths) may omit stride/size metadata.
        // For our single-plane ARGB/XRGB capture path, derive conservative defaults.
        if modifier == 0 && stride == 0 && width != 0 {
            stride = width.saturating_mul(4);
        }
        let size_bytes = (u64::from(offset) + (u64::from(stride).saturating_mul(u64::from(height))))
            .min(u64::from(u32::MAX)) as u32;

        debug!(
            "export frame (passthrough): modifier=0x{:016x} fourcc=0x{:08x} stride={} offset={} size_bytes={}",
            modifier,
            self.drm_fourcc,
            stride,
            offset,
            size_bytes
        );

        let (acquire, acquire_fd) = self.create_exportable_semaphore()?;
        let (release, release_fd) = self.create_exportable_semaphore()?;

        let mut imported_wait_sem: Option<vk::Semaphore> = None;
        if let Some(fd) = wait_fd.0.take() {
            match self.import_sync_fd_semaphore(fd) {
                Ok(sem) => {
                    imported_wait_sem = Some(sem);
                }
                Err(err) => {
                    warn!("failed to import acquire sync fd into semaphore (passthrough): {err}");
                    let _ = close(fd);
                }
            }
        }

        // Submit a trivial no-op command buffer that waits on previous consumer release (backpressure)
        // and signals acquire. We intentionally submit a real command buffer (even empty) because
        // some driver stacks appear to produce external semaphore handles that CUDA can import
        // reliably only after being used in a queue submission with at least one command buffer.
        // Passthrough mode does not strictly depend on consumer release fences. Waiting on them
        // can deadlock the compositor if the consumer misses a single fence signal (e.g. CUDA import hiccup).
        // Keep the producer running and rely on the fixed-FPS pacing + slot fences for backpressure.
        let mut wait_sems: Vec<vk::Semaphore> = Vec::new();
        let mut wait_stages: Vec<vk::PipelineStageFlags> = Vec::new();
        if let Some(wait_sem) = imported_wait_sem {
            wait_sems.push(wait_sem);
            wait_stages.push(vk::PipelineStageFlags::TRANSFER);
        }

        let cmd = self.slots[slot_idx].cmd;
        unsafe { self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }.ok();
        unsafe {
            self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
            self.device.end_command_buffer(cmd)?;
        }

        let signal_sems = [acquire];
        let cmd_bufs = [cmd];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&cmd_bufs)
            .signal_semaphores(&signal_sems);
        unsafe { self.device.queue_submit(self.queue, &[submit], self.slots[slot_idx].fence) }.context("queue_submit")?;

        if let Some(buf) = buffer_hold {
            let slot = &mut self.slots[slot_idx];
            slot.to_release_buffers.push(buf);
        }

        if let Some(wait_sem) = imported_wait_sem {
            let slot = &mut self.slots[slot_idx];
            slot.to_destroy_sems.push(wait_sem);
        }

        self.frame_id += 1;
        {
            let slot = &mut self.slots[slot_idx];
            if let (Some(pa), Some(pr)) = (slot.prev_acquire.take(), slot.prev_release.take()) {
                slot.to_destroy.push((pa, pr));
            }
            slot.prev_acquire = Some(acquire);
            slot.prev_release = Some(release);
        }

        Ok(RenderedFrame {
            frame_id: self.frame_id,
            width,
            height,
            drm_fourcc: self.drm_fourcc,
            modifier,
            stride,
            offset,
            size_bytes,
            dmabuf_fd,
            acquire_fence_fd: acquire_fd,
            release_fence_fd: release_fd,
        })
    }

    fn import_dmabuf_image(
        &self,
        src: &Dmabuf,
    ) -> anyhow::Result<(vk::Image, vk::DeviceMemory, u32, u32, vk::ImageLayout)> {
        if src.num_planes() != 1 {
            anyhow::bail!("only single-plane dmabufs are supported (got {} planes)", src.num_planes());
        }

        let fmt = src.format().code;
        let vk_format = match fmt {
            ShmFourcc::Argb8888 | ShmFourcc::Xrgb8888 => vk::Format::B8G8R8A8_UNORM,
            other => anyhow::bail!("unsupported source dmabuf format {other:?} (supported: ARGB8888/XRGB8888)"),
        };

        let size = src.size();
        let width = size.w as u32;
        let height = size.h as u32;

        let modifier_u64: u64 = src.format().modifier.into();
        let offset = src.offsets().next().unwrap_or(0) as u64;
        let stride = src.strides().next().unwrap_or(0) as u64;

        // Allow opting into an OPTIMAL-tiling DMA-BUF import path, which lets the driver interpret
        // the underlying memory layout without a DRM modifier contract (useful for wl_drm PRIME
        // buffers which do not carry modifiers).
        //
        // Note: When modifier=0 (DRM_FORMAT_MOD_LINEAR), some producer stacks still fail to provide
        // reliable modifier metadata. In that situation, the LINEAR-tiling Vulkan import can succeed
        // but produce stable "banding"/garbage. This flag allows forcing the OPAQUE_FD path even
        // when the linear import would otherwise be chosen.
        let prefer_opaque_optimal = std::env::var("METABONK_EYE_IMPORT_OPAQUE_OPTIMAL")
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(false);
        if prefer_opaque_optimal {
            match self.import_dmabuf_image_opaque_optimal(src, vk_format, width, height) {
                Ok((image, memory)) => {
                    info!(
                        "dmabuf image import: using DMA_BUF_EXT + OPTIMAL tiling (no modifier contract)"
                    );
                    return Ok((image, memory, width, height, vk::ImageLayout::PREINITIALIZED));
                }
                Err(e) => {
                    warn!("DMA_BUF_EXT + OPTIMAL dmabuf import failed; falling back to other import modes: {e}");
                }
            }
        }

        // Fast path for linear buffers: attempt a LINEAR tiling import first (avoid
        // VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT for modifier=0, which has produced persistent
        // banding on some driver stacks).
        let force_drm_mod_import = std::env::var("METABONK_EYE_FORCE_DRM_MOD_IMPORT")
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(false);
        if modifier_u64 == 0 && !force_drm_mod_import {
            match self.import_dmabuf_image_linear(src, vk_format, width, height) {
                Ok((image, memory)) => {
                    info!("dmabuf image import: using LINEAR tiling path for modifier=0");
                    return Ok((image, memory, width, height, vk::ImageLayout::PREINITIALIZED));
                }
                Err(e) => {
                    warn!("linear-tiling dmabuf import failed; falling back to DRM modifier import: {e}");
                }
            }
        }

        let src_fd = src.handles().next().unwrap();

        // Some drivers require a non-zero `size` for explicit modifier images.
        // Prefer the dmabuf file size (when available) to avoid under-describing the plane for
        // tiled/compressed allocations that still report a "linear-looking" stride/modifier.
        let mut plane_size = stride.saturating_mul(height as u64);
        unsafe {
            let mut st: libc::stat = std::mem::zeroed();
            if libc::fstat(src_fd.as_raw_fd(), &mut st) == 0 && st.st_size > 0 {
                plane_size = plane_size.max(st.st_size as u64);
            }
        }
        let plane_layout = vk::SubresourceLayout {
            offset,
            size: plane_size,
            row_pitch: stride,
            array_pitch: 0,
            depth_pitch: 0,
        };
        let fd0 = unsafe { libc::dup(src_fd.as_raw_fd()) };
        if fd0 < 0 {
            return Err(std::io::Error::last_os_error()).context("dup dmabuf fd");
        }

        let try_import = |handle_type: vk::ExternalMemoryHandleTypeFlags, fd: i32| -> anyhow::Result<(vk::Image, vk::DeviceMemory)> {
            let mut explicit =
                vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                    .drm_format_modifier(modifier_u64)
                    .plane_layouts(std::slice::from_ref(&plane_layout));
            let mut external =
                vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_type);

            let mut image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk_format)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                // Imported modifier-backed DMA-BUFs already contain valid pixels. Using UNDEFINED
                // permits the driver to discard contents on the first transition.
                .initial_layout(IMAGE_LAYOUT_DRM_FORMAT_MODIFIER_OPTIMAL_EXT);
            image_info = image_info.push_next(&mut external).push_next(&mut explicit);

            let image = unsafe { self.device.create_image(&image_info, None) }.context("create imported image")?;

            let mut dedicated_req = vk::MemoryDedicatedRequirements::default();
            let mut mem_req2 = vk::MemoryRequirements2::default().push_next(&mut dedicated_req);
            let req_info2 = vk::ImageMemoryRequirementsInfo2::default().image(image);
            unsafe { self.device.get_image_memory_requirements2(&req_info2, &mut mem_req2) };
            let mem_req = mem_req2.memory_requirements;

            let mut fd_props = vk::MemoryFdPropertiesKHR::default();
            if let Err(e) = unsafe { self.khr_mem_fd.get_memory_fd_properties(handle_type, fd, &mut fd_props) } {
                unsafe { self.device.destroy_image(image, None) };
                let _ = unsafe { libc::close(fd) };
                return Err(anyhow::anyhow!(e)).context("vkGetMemoryFdPropertiesKHR");
            }
            let allowed_type_bits = mem_req.memory_type_bits & fd_props.memory_type_bits;
            if allowed_type_bits == 0 {
                unsafe { self.device.destroy_image(image, None) };
                let _ = unsafe { libc::close(fd) };
                anyhow::bail!(
                    "no compatible Vulkan memory types for imported fd (handle={handle_type:?} mem_req_bits=0x{:x} fd_bits=0x{:x})",
                    mem_req.memory_type_bits,
                    fd_props.memory_type_bits
                );
            }

            let mut import_info = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(handle_type)
                .fd(fd);
            let mut dedicated_alloc = vk::MemoryDedicatedAllocateInfo::default().image(image);
            let mut alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_req.size)
                .memory_type_index(select_best_memory_type_bits(
                    self.instance.handle(),
                    self.physical.handle(),
                    allowed_type_bits,
                )?);
            if dedicated_req.requires_dedicated_allocation == vk::TRUE || dedicated_req.prefers_dedicated_allocation == vk::TRUE {
                alloc = alloc.push_next(&mut dedicated_alloc);
            }
            let alloc = alloc.push_next(&mut import_info);
            let memory = match unsafe { self.device.allocate_memory(&alloc, None) } {
                Ok(m) => m,
                Err(e) => {
                    unsafe { self.device.destroy_image(image, None) };
                    // On import allocation failure Vulkan did not take ownership of `fd`.
                    let _ = unsafe { libc::close(fd) };
                    let best = describe_memory_type_bits(self.instance.handle(), self.physical.handle(), allowed_type_bits);
                    anyhow::bail!(
                        "allocate imported memory failed: {e:?} (handle={handle_type:?} src={}x{} stride={} offset={} modifier=0x{:016x} mem_req.size={} allowed_types=0x{:x} {})",
                        width,
                        height,
                        stride,
                        offset,
                        modifier_u64,
                        mem_req.size,
                        allowed_type_bits,
                        best
                    );
                }
            };
            unsafe { self.device.bind_image_memory(image, memory, 0) }.context("bind imported image memory")?;
            Ok((image, memory))
        };

        // Prefer DMA_BUF_EXT when possible (true dma-buf import). Some driver stacks surface opaque-fd
        // handle types only; fall back to OPAQUE_FD when DMA_BUF_EXT import fails.
        let (image, memory) = match try_import(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT, fd0) {
            Ok(v) => v,
            Err(e_dmabuf) => {
                warn!("DMA_BUF_EXT import failed; retrying as OPAQUE_FD: {e_dmabuf}");
                let fd1 = unsafe { libc::dup(src_fd.as_raw_fd()) };
                if fd1 < 0 {
                    return Err(std::io::Error::last_os_error()).context("dup dmabuf fd (retry opaque)");
                }
                try_import(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD, fd1)
                    .with_context(|| format!("OPAQUE_FD import also failed (after DMA_BUF_EXT): {e_dmabuf}"))?
            }
        };

        Ok((
            image,
            memory,
            width,
            height,
            IMAGE_LAYOUT_DRM_FORMAT_MODIFIER_OPTIMAL_EXT,
        ))
    }

    fn import_dmabuf_image_linear(
        &self,
        src: &Dmabuf,
        vk_format: vk::Format,
        width: u32,
        height: u32,
    ) -> anyhow::Result<(vk::Image, vk::DeviceMemory)> {
        let src_fd = src.handles().next().unwrap();
        let fd0 = unsafe { libc::dup(src_fd.as_raw_fd()) };
        if fd0 < 0 {
            return Err(std::io::Error::last_os_error()).context("dup dmabuf fd (linear)");
        }

        let handle_type = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_type);

        let mut image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            // The dma-buf contains producer-written pixels at import time; PREINITIALIZED ensures
            // the first transition preserves contents (UNDEFINED allows discard).
            .initial_layout(vk::ImageLayout::PREINITIALIZED);
        image_info = image_info.push_next(&mut external);

        let image = unsafe { self.device.create_image(&image_info, None) }.context("create imported linear image")?;

        let mut dedicated_req = vk::MemoryDedicatedRequirements::default();
        let mut mem_req2 = vk::MemoryRequirements2::default().push_next(&mut dedicated_req);
        let req_info2 = vk::ImageMemoryRequirementsInfo2::default().image(image);
        unsafe { self.device.get_image_memory_requirements2(&req_info2, &mut mem_req2) };
        let mem_req = mem_req2.memory_requirements;

        let mut fd_props = vk::MemoryFdPropertiesKHR::default();
        if let Err(e) = unsafe { self.khr_mem_fd.get_memory_fd_properties(handle_type, fd0, &mut fd_props) } {
            unsafe { self.device.destroy_image(image, None) };
            let _ = unsafe { libc::close(fd0) };
            return Err(anyhow::anyhow!(e)).context("vkGetMemoryFdPropertiesKHR(linear)");
        }
        let allowed_type_bits = mem_req.memory_type_bits & fd_props.memory_type_bits;
        if allowed_type_bits == 0 {
            unsafe { self.device.destroy_image(image, None) };
            let _ = unsafe { libc::close(fd0) };
            anyhow::bail!(
                "no compatible Vulkan memory types for imported fd (linear mem_req_bits=0x{:x} fd_bits=0x{:x})",
                mem_req.memory_type_bits,
                fd_props.memory_type_bits
            );
        }

        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(handle_type)
            .fd(fd0);
        let mut dedicated_alloc = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let mut alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(select_best_memory_type_bits(
                self.instance.handle(),
                self.physical.handle(),
                allowed_type_bits,
            )?);
        if dedicated_req.requires_dedicated_allocation == vk::TRUE || dedicated_req.prefers_dedicated_allocation == vk::TRUE {
            alloc = alloc.push_next(&mut dedicated_alloc);
        }
        let alloc = alloc.push_next(&mut import_info);
        let memory = match unsafe { self.device.allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe { self.device.destroy_image(image, None) };
                // On import allocation failure Vulkan did not take ownership of `fd0`.
                let _ = unsafe { libc::close(fd0) };
                let best = describe_memory_type_bits(self.instance.handle(), self.physical.handle(), allowed_type_bits);
                anyhow::bail!(
                    "allocate imported memory failed (linear): {e:?} (src={}x{} mem_req.size={} allowed_types=0x{:x} {})",
                    width,
                    height,
                    mem_req.size,
                    allowed_type_bits,
                    best
                );
            }
        };
        unsafe { self.device.bind_image_memory(image, memory, 0) }.context("bind imported linear image memory")?;

        // Validate that the driver's linear image layout matches the dma-buf plane metadata.
        // If the row pitch or offset do not match, Vulkan will interpret the imported memory
        // with a different stride than the producer used, producing stable "banding"/garbage.
        let expected_stride = {
            let stride = src.strides().next().unwrap_or(0) as u64;
            if stride == 0 {
                (width as u64).saturating_mul(4)
            } else {
                stride
            }
        };
        let expected_offset = src.offsets().next().unwrap_or(0) as u64;
        let sub = vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::COLOR);
        let layout = unsafe { self.device.get_image_subresource_layout(image, sub) };
        if layout.row_pitch != expected_stride || layout.offset != expected_offset {
            unsafe {
                self.device.destroy_image(image, None);
                self.device.free_memory(memory, None);
            }
            anyhow::bail!(
                "imported linear image layout mismatch (expected offset={} row_pitch={}, got offset={} row_pitch={}): refusing LINEAR import",
                expected_offset,
                expected_stride,
                layout.offset,
                layout.row_pitch
            );
        }

        Ok((image, memory))
    }

    fn import_dmabuf_image_opaque_optimal(
        &self,
        src: &Dmabuf,
        vk_format: vk::Format,
        width: u32,
        height: u32,
    ) -> anyhow::Result<(vk::Image, vk::DeviceMemory)> {
        let src_fd = src.handles().next().unwrap();
        let src_raw_fd = src_fd.as_raw_fd();

        // wl_drm PRIME buffers do not convey DRM modifiers. For those, importing with
        // VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT + OPTIMAL tiling lets the driver interpret
        // the underlying layout without a modifier contract (and avoids treating "modifier=0" as
        // strictly LINEAR when it may be unknown).
        let handle_type = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_type);

        let mut image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            // Imported DMA-BUFs already contain valid pixels. Using UNDEFINED permits the
            // driver to discard contents on the first transition. PREINITIALIZED preserves contents
            // so we can transition into GENERAL for transfer without losing the frame.
            .initial_layout(vk::ImageLayout::PREINITIALIZED);
        image_info = image_info.push_next(&mut external);

        let image = unsafe { self.device.create_image(&image_info, None) }.context("create imported opaque-optimal image")?;

        let mut dedicated_req = vk::MemoryDedicatedRequirements::default();
        let mut mem_req2 = vk::MemoryRequirements2::default().push_next(&mut dedicated_req);
        let req_info2 = vk::ImageMemoryRequirementsInfo2::default().image(image);
        unsafe { self.device.get_image_memory_requirements2(&req_info2, &mut mem_req2) };
        let mem_req = mem_req2.memory_requirements;

        let mut allowed_type_bits = mem_req.memory_type_bits;
        let fd_for_props = unsafe { libc::dup(src_raw_fd) };
        if fd_for_props >= 0 {
            let mut fd_props = vk::MemoryFdPropertiesKHR::default();
            match unsafe { self.khr_mem_fd.get_memory_fd_properties(handle_type, fd_for_props, &mut fd_props) } {
                Ok(()) => {
                    allowed_type_bits &= fd_props.memory_type_bits;
                }
                Err(e) => {
                    // Some driver stacks reject querying OPAQUE_FD properties for dma-buf-backed
                    // fds. We can still attempt to import by trying memory types from the image
                    // requirements and letting vkAllocateMemory validate the handle.
                    warn!("vkGetMemoryFdPropertiesKHR(opaque_optimal) failed ({e:?}); attempting import without fd properties");
                }
            }
            let _ = unsafe { libc::close(fd_for_props) };
        }
        if allowed_type_bits == 0 {
            unsafe { self.device.destroy_image(image, None) };
            anyhow::bail!(
                "no compatible Vulkan memory types for imported fd (opaque-optimal mem_req_bits=0x{:x})",
                mem_req.memory_type_bits
            );
        }

        let preferred = select_best_memory_type_bits(self.instance.handle(), self.physical.handle(), allowed_type_bits)?;
        let mut candidates: Vec<u32> = Vec::new();
        candidates.push(preferred);
        for idx in 0..32u32 {
            if idx == preferred {
                continue;
            }
            if (allowed_type_bits & (1u32 << idx)) != 0 {
                candidates.push(idx);
            }
        }

        let mut last_err: Option<vk::Result> = None;
        for idx in candidates {
            let fd_try = unsafe { libc::dup(src_raw_fd) };
            if fd_try < 0 {
                last_err = Some(vk::Result::ERROR_INVALID_EXTERNAL_HANDLE);
                break;
            }
            let mut import_info = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(handle_type)
                .fd(fd_try);
            let mut dedicated_alloc = vk::MemoryDedicatedAllocateInfo::default().image(image);
            let mut alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_req.size)
                .memory_type_index(idx);
            if dedicated_req.requires_dedicated_allocation == vk::TRUE || dedicated_req.prefers_dedicated_allocation == vk::TRUE {
                alloc = alloc.push_next(&mut dedicated_alloc);
            }
            let alloc = alloc.push_next(&mut import_info);
            match unsafe { self.device.allocate_memory(&alloc, None) } {
                Ok(memory) => {
                    // Vulkan owns fd_try after successful import allocation.
                    unsafe { self.device.bind_image_memory(image, memory, 0) }
                        .context("bind imported opaque-optimal image memory")?;
                    return Ok((image, memory));
                }
                Err(e) => {
                    // On import allocation failure Vulkan did not take ownership of `fd_try`.
                    let _ = unsafe { libc::close(fd_try) };
                    last_err = Some(e);
                }
            }
        }

        unsafe { self.device.destroy_image(image, None) };
        anyhow::bail!("allocate imported memory failed (opaque-optimal): {last_err:?}")
    }

    fn import_dmabuf_buffer(
        &self,
        src: &Dmabuf,
    ) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory, u32, u32, u64, u64)> {
        if src.num_planes() != 1 {
            anyhow::bail!("only single-plane dmabufs are supported (got {} planes)", src.num_planes());
        }

        let fmt = src.format().code;
        match fmt {
            ShmFourcc::Argb8888 | ShmFourcc::Xrgb8888 => {}
            other => anyhow::bail!("unsupported source dmabuf format {other:?} (supported: ARGB8888/XRGB8888)"),
        };

        let size = src.size();
        let width = size.w as u32;
        let height = size.h as u32;
        let offset = src.offsets().next().unwrap_or(0) as u64;
        let stride = src.strides().next().unwrap_or(0) as u64;
        let row_pitch = if stride == 0 {
            (width as u64) * 4
        } else {
            stride
        };
        let size_bytes = offset + row_pitch.saturating_mul(height as u64);

        let src_fd = src.handles().next().unwrap();
        let fd0 = unsafe { libc::dup(src_fd.as_raw_fd()) };
        if fd0 < 0 {
            return Err(std::io::Error::last_os_error()).context("dup dmabuf fd");
        }

        let try_import = |handle_type: vk::ExternalMemoryHandleTypeFlags, fd: i32| -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
            let mut external =
                vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
            let mut buf_info = vk::BufferCreateInfo::default()
                .size(size_bytes)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            buf_info = buf_info.push_next(&mut external);

            let buffer = unsafe { self.device.create_buffer(&buf_info, None) }.context("create imported buffer")?;

            let mut dedicated_req = vk::MemoryDedicatedRequirements::default();
            let mut mem_req2 = vk::MemoryRequirements2::default().push_next(&mut dedicated_req);
            let req_info2 = vk::BufferMemoryRequirementsInfo2::default().buffer(buffer);
            unsafe { self.device.get_buffer_memory_requirements2(&req_info2, &mut mem_req2) };
            let mem_req = mem_req2.memory_requirements;

            let mut fd_props = vk::MemoryFdPropertiesKHR::default();
            if let Err(e) = unsafe { self.khr_mem_fd.get_memory_fd_properties(handle_type, fd, &mut fd_props) } {
                unsafe { self.device.destroy_buffer(buffer, None) };
                let _ = unsafe { libc::close(fd) };
                return Err(anyhow::anyhow!(e)).context("vkGetMemoryFdPropertiesKHR(buffer)");
            }
            let allowed_type_bits = mem_req.memory_type_bits & fd_props.memory_type_bits;
            if allowed_type_bits == 0 {
                unsafe { self.device.destroy_buffer(buffer, None) };
                let _ = unsafe { libc::close(fd) };
                anyhow::bail!(
                    "no compatible Vulkan memory types for imported fd (buffer handle={handle_type:?} mem_req_bits=0x{:x} fd_bits=0x{:x})",
                    mem_req.memory_type_bits,
                    fd_props.memory_type_bits
                );
            }

            let mut import_info = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(handle_type)
                .fd(fd);
            let mut dedicated_alloc = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);

            let mut candidates: Vec<(u32, bool, u64)> = Vec::new();
            {
                let props = unsafe { self.instance.handle().get_physical_device_memory_properties(self.physical.handle()) };
                for i in 0..props.memory_type_count {
                    if (allowed_type_bits & (1 << i)) == 0 {
                        continue;
                    }
                    let mt = props.memory_types[i as usize];
                    let flags = mt.property_flags;
                    let is_device_local = flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
                    let heap_idx = mt.heap_index as usize;
                    let heap_size = props.memory_heaps.get(heap_idx).map(|h| h.size).unwrap_or(0);
                    candidates.push((i, is_device_local, heap_size));
                }
            }
            candidates.sort_by(|a, b| {
                b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)).then_with(|| a.0.cmp(&b.0))
            });

            let best = describe_memory_type_bits(self.instance.handle(), self.physical.handle(), allowed_type_bits);
            let mut last_err: Option<vk::Result> = None;
            let mut memory: Option<vk::DeviceMemory> = None;
            for (mem_type_idx, _dev, _heap) in candidates {
                let mut alloc = vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_req.size)
                    .memory_type_index(mem_type_idx);
                if dedicated_req.requires_dedicated_allocation == vk::TRUE || dedicated_req.prefers_dedicated_allocation == vk::TRUE {
                    alloc = alloc.push_next(&mut dedicated_alloc);
                }
                let alloc = alloc.push_next(&mut import_info);
                match unsafe { self.device.allocate_memory(&alloc, None) } {
                    Ok(m) => {
                        memory = Some(m);
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e);
                        continue;
                    }
                }
            }
            let memory = match memory {
                Some(m) => m,
                None => {
                    unsafe { self.device.destroy_buffer(buffer, None) };
                    let _ = unsafe { libc::close(fd) };
                    anyhow::bail!(
                        "allocate imported buffer memory failed: {:?} (handle={handle_type:?} src={}x{} stride={} offset={} mem_req.size={} allowed_types=0x{:x} {})",
                        last_err.unwrap_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
                        width,
                        height,
                        row_pitch,
                        offset,
                        mem_req.size,
                        allowed_type_bits,
                        best
                    );
                }
            };
            unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }.context("bind imported buffer memory")?;
            Ok((buffer, memory))
        };

        let (buffer, memory) = match try_import(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT, fd0) {
            Ok(v) => v,
            Err(e_dmabuf) => {
                warn!("DMA_BUF_EXT buffer import failed; retrying as OPAQUE_FD: {e_dmabuf}");
                let fd1 = unsafe { libc::dup(src_fd.as_raw_fd()) };
                if fd1 < 0 {
                    return Err(std::io::Error::last_os_error()).context("dup dmabuf fd (retry opaque buffer)");
                }
                try_import(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD, fd1)
                    .with_context(|| format!("OPAQUE_FD buffer import also failed (after DMA_BUF_EXT): {e_dmabuf}"))?
            }
        };

        Ok((buffer, memory, width, height, row_pitch, offset))
    }

    pub fn on_consumer_disconnect(&mut self) {
        for slot in &mut self.slots {
            if let (Some(pa), Some(pr)) = (slot.prev_acquire.take(), slot.prev_release.take()) {
                slot.to_destroy.push((pa, pr));
            }
            slot.prev_acquire = None;
            slot.prev_release = None;
        }
    }

    pub fn drop_cached_source(&mut self) {
        if let Some(c) = self.last_src.take() {
            unsafe {
                match c.kind {
                    SrcKind::Image { image, memory, .. } => {
                        self.device.destroy_image(image, None);
                        self.device.free_memory(memory, None);
                    }
                    SrcKind::Buffer { buffer, memory, .. } => {
                        self.device.destroy_buffer(buffer, None);
                        self.device.free_memory(memory, None);
                    }
                }
            }
        }
    }

    fn wait_for_fence(&self, fence: vk::Fence, label: &str, slot_idx: usize) -> anyhow::Result<()> {
        // Avoid indefinite hangs. If the GPU queue wedges (e.g. external semaphore never signals),
        // we prefer a fast failure so the supervisor can restart the compositor.
        // Default to 5s (configurable via env) so stalls are surfaced with logs.
        let default_wait_ms = 5_000;
        let wait_ms = std::env::var("METABONK_EYE_FENCE_WAIT_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(default_wait_ms)
            .max(default_wait_ms);
        let wait_ns = wait_ms.saturating_mul(1_000_000);
        info!(
            "⏳ Waiting for {} fence (slot={}, timeout_ms={})...",
            label,
            slot_idx,
            wait_ms
        );
        match unsafe { self.device.wait_for_fences(&[fence], true, wait_ns) } {
            Ok(_) => {
                info!("✅ {} fence signaled (slot={})", label, slot_idx);
                Ok(())
            }
            Err(vk::Result::TIMEOUT) => {
                warn!(
                    "{} fence wait timed out (slot={}, timeout_ms={})",
                    label,
                    slot_idx,
                    wait_ms
                );
                anyhow::bail!("slot fence wait timed out (GPU queue stalled)")
            }
            Err(e) => Err(anyhow::anyhow!("{} fence wait failed (slot={}): {e:?}", label, slot_idx)),
        }
    }

    fn debug_dump_buffer_ppm(&self, memory: vk::DeviceMemory, size: u64, frame_id: u64) -> anyhow::Result<()> {
        let row_bytes = (self.width as usize).saturating_mul(4);
        self.debug_dump_buffer_ppm_kind(memory, size, row_bytes, frame_id, "staging")
    }

    fn debug_dump_buffer_ppm_kind(
        &self,
        memory: vk::DeviceMemory,
        size: u64,
        row_pitch_bytes: usize,
        frame_id: u64,
        kind: &str,
    ) -> anyhow::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let width = self.width as usize;
        let height = self.height as usize;
        let packed_row_bytes = width.saturating_mul(4);
        if row_pitch_bytes < packed_row_bytes {
            anyhow::bail!(
                "debug {kind} row pitch too small (row_pitch_bytes={row_pitch_bytes}, packed_row_bytes={packed_row_bytes})"
            );
        }
        let needed = (row_pitch_bytes as u64).saturating_mul(self.height as u64);
        if size < needed {
            anyhow::bail!("debug {kind} buffer too small (size={size}, needed={needed})");
        }

        let ptr = unsafe { self.device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) }
            .context("map debug buffer")?;
        let data = unsafe { std::slice::from_raw_parts(ptr as *const u8, size as usize) };

        let ppm_path = format!("/tmp/debug_{kind}_{frame_id}.ppm");
        let mut f = File::create(&ppm_path).context("create debug ppm")?;
        writeln!(f, "P6")?;
        writeln!(f, "{} {}", self.width, self.height)?;
        writeln!(f, "255")?;

        let total_pixels: u64 = (width as u64).saturating_mul(height as u64);
        let mut sum_luma: u128 = 0;
        let mut sumsq_luma: u128 = 0;
        let mut black_pixels: u64 = 0;
        let mut max_luma: u8 = 0;
        let sample_step: usize = 8;
        let mut unique_sample: HashSet<u32> = HashSet::new();
        let mut sampled_pixels: u64 = 0;
        for y in 0..height {
            let row_offset = y * row_pitch_bytes;
            let end = row_offset.saturating_add(packed_row_bytes);
            if end > data.len() {
                break;
            }
            let row = &data[row_offset..end];
            for x in 0..width {
                let px = x * 4;
                let b = row[px];
                let g = row[px + 1];
                let r = row[px + 2];
                let luma_u16 = ((r as u16) + (g as u16) + (b as u16)) / 3;
                let luma = luma_u16 as u8;
                sum_luma = sum_luma.saturating_add(luma_u16 as u128);
                sumsq_luma = sumsq_luma.saturating_add((luma_u16 as u128).saturating_mul(luma_u16 as u128));
                if r == 0 && g == 0 && b == 0 {
                    black_pixels = black_pixels.saturating_add(1);
                }
                max_luma = max_luma.max(luma);
                if (x % sample_step == 0) && (y % sample_step == 0) {
                    let packed = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
                    unique_sample.insert(packed);
                    sampled_pixels = sampled_pixels.saturating_add(1);
                }
                f.write_all(&[r, g, b])?;
            }
        }

        unsafe { self.device.unmap_memory(memory) };
        let n = total_pixels.max(1) as f64;
        let mean_luma = (sum_luma as f64) / n;
        let var_luma = ((sumsq_luma as f64) / n) - (mean_luma * mean_luma);
        let std_luma = var_luma.max(0.0).sqrt();
        let black_frac = (black_pixels as f64) / n;
        info!(
            frame_id,
            mean_luma,
            std_luma,
            black_frac,
            max_luma,
            unique_colors_sampled = unique_sample.len(),
            sampled_pixels,
            sample_step,
            "{kind} dump stats: wrote {} (convert with: convert {} /tmp/debug_{kind}_{frame_id}.png)",
            ppm_path,
            ppm_path
        );
        Ok(())
    }

    fn should_debug_dump_staging(&self, frame_id: u64, use_linear_export: bool) -> bool {
        if !self.debug_dump_staging || self.debug_dump_done || !use_linear_export {
            return false;
        }
        let after_frame = self.debug_dump_after_frame.unwrap_or(1);
        if frame_id < after_frame {
            return false;
        }
        if let Some(max) = self.debug_dump_max {
            if self.debug_dump_count >= max {
                return false;
            }
        }
        if let Some(every_n) = self.debug_dump_every_n {
            let delta = frame_id.saturating_sub(after_frame);
            return delta % every_n == 0;
        }
        true
    }

    fn export_memory_dmabuf_fd(&self, memory: vk::DeviceMemory) -> anyhow::Result<RawFd> {
        let info = vk::MemoryGetFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
            .memory(memory);
        let fd = unsafe { self.khr_mem_fd.get_memory_fd(&info) }.context("vkGetMemoryFdKHR")?;
        Ok(fd)
    }

    fn create_exportable_semaphore(&self) -> anyhow::Result<(vk::Semaphore, RawFd)> {
        let mut export =
            vk::ExportSemaphoreCreateInfo::default().handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let sem = unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut export), None) }
        .context("create_semaphore")?;
        let fd_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(sem)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let fd = unsafe { self.khr_sema_fd.get_semaphore_fd(&fd_info) }.context("vkGetSemaphoreFdKHR")?;
        Ok((sem, fd))
    }

    fn import_sync_fd_semaphore(&self, fd: RawFd) -> anyhow::Result<vk::Semaphore> {
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let sem = unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut export), None) }
            .context("create sync-fd semaphore")?;
        let import_info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(sem)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
            .flags(vk::SemaphoreImportFlags::TEMPORARY)
            .fd(fd);
        unsafe { self.khr_sema_fd.import_semaphore_fd(&import_info) }
            .context("vkImportSemaphoreFdKHR")?;
        Ok(sem)
    }

    fn create_sync_fd_export_semaphore(&self) -> anyhow::Result<vk::Semaphore> {
        let mut export =
            vk::ExportSemaphoreCreateInfo::default().handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut export), None) }
            .context("create sync-fd export semaphore")
    }

    fn export_semaphore_sync_fd(&self, sem: vk::Semaphore) -> anyhow::Result<RawFd> {
        let fd_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(sem)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let fd = unsafe { self.khr_sema_fd.get_semaphore_fd(&fd_info) }.context("vkGetSemaphoreFdKHR(sync_fd)")?;
        Ok(fd)
    }

    fn import_dmabuf_read_fence(&self, dmabuf_fd: RawFd, fence_fd: RawFd) -> anyhow::Result<()> {
        let arg = DmaBufImportSyncFile {
            flags: DMA_BUF_SYNC_READ,
            fd: fence_fd as i32,
        };
        // SAFETY: ioctl reads the pointed struct and takes a reference to the provided sync_file fd.
        // The caller remains responsible for closing `fence_fd`.
        unsafe { dma_buf_import_sync_file(dmabuf_fd, &arg) }.context("DMA_BUF_IOCTL_IMPORT_SYNC_FILE")?;
        Ok(())
    }
}

impl Drop for VulkanProducer {
    fn drop(&mut self) {
        unsafe {
            if let Some(c) = self.last_src.take() {
                match c.kind {
                    SrcKind::Image { image, memory, .. } => {
                        self.device.destroy_image(image, None);
                        self.device.free_memory(memory, None);
                    }
                    SrcKind::Buffer { buffer, memory, .. } => {
                        self.device.destroy_buffer(buffer, None);
                        self.device.free_memory(memory, None);
                    }
                }
            }
            for slot in self.slots.drain(..) {
                for res in slot.to_destroy_imported {
                    match res {
                        ImportedResource::Image(img, mem) => {
                            self.device.destroy_image(img, None);
                            self.device.free_memory(mem, None);
                        }
                        ImportedResource::Buffer(buf, mem) => {
                            self.device.destroy_buffer(buf, None);
                            self.device.free_memory(mem, None);
                        }
                    }
                }
                for (a, r) in slot.to_destroy {
                    self.device.destroy_semaphore(a, None);
                    self.device.destroy_semaphore(r, None);
                }
                for sem in slot.to_destroy_sems {
                    self.device.destroy_semaphore(sem, None);
                }
                self.device.destroy_semaphore(slot.copy_done, None);
                if let Some(a) = slot.prev_acquire {
                    self.device.destroy_semaphore(a, None);
                }
                if let Some(r) = slot.prev_release {
                    self.device.destroy_semaphore(r, None);
                }
                self.device.destroy_fence(slot.fence, None);
                if let Some(buf) = slot.linear_buffer {
                    self.device.destroy_buffer(buf, None);
                }
                if let Some(mem) = slot.linear_memory {
                    self.device.free_memory(mem, None);
                }
                self.device.destroy_image(slot.image, None);
                self.device.free_memory(slot.memory, None);
            }
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
        }
    }
}

pub struct RenderedFrame {
    pub frame_id: u64,
    pub width: u32,
    pub height: u32,
    pub drm_fourcc: u32,
    pub modifier: u64,
    pub stride: u32,
    pub offset: u32,
    pub size_bytes: u32,
    pub dmabuf_fd: RawFd,
    pub acquire_fence_fd: RawFd,
    pub release_fence_fd: RawFd,
}

impl RenderedFrame {
    pub fn close_fds(&self) {
        let _ = close(self.dmabuf_fd);
        let _ = close(self.acquire_fence_fd);
        let _ = close(self.release_fence_fd);
    }
}

fn clear_color(phase: u8) -> vk::ClearColorValue {
    let t = (phase as f32) / 255.0;
    vk::ClearColorValue {
        float32: [t, 0.2, 1.0 - t, 1.0],
    }
}

fn pick_physical_device(instance: &Instance, select: &VkSelect) -> anyhow::Result<PhysicalDevice> {
    let mut devices: Vec<PhysicalDevice> = PhysicalDevice::enumerate(instance)
        .context("enumerate physical devices")?
        .collect();
    if devices.is_empty() {
        anyhow::bail!("no Vulkan physical devices found");
    }

    if let Some(name) = select.device_name_contains.as_ref().map(|s| s.to_ascii_lowercase()) {
        if let Some(pos) = devices
            .iter()
            .position(|d| d.name().to_ascii_lowercase().contains(&name))
        {
            let chosen = devices.remove(pos);
            info!(
                "selected Vulkan device by name: {} (api={:?})",
                chosen.name(),
                chosen.api_version()
            );
            return Ok(chosen);
        }
        anyhow::bail!("no Vulkan device name matched '{name}'");
    }

    if let Some(idx) = select.device_index {
        let idx = idx as usize;
        if idx >= devices.len() {
            anyhow::bail!("vk-device-index {idx} out of range ({} devices)", devices.len());
        }
        let chosen = devices.remove(idx);
        info!(
            "selected Vulkan device by index: {} (api={:?})",
            chosen.name(),
            chosen.api_version()
        );
        return Ok(chosen);
    }

    // Default: prefer a device with a DRM render node if available, else fallback to first.
    devices.sort_by_key(|d| d.render_node().ok().flatten().is_none());
    let chosen = devices.remove(0);
    info!(
        "selected Vulkan device: {} (api={:?})",
        chosen.name(),
        chosen.api_version()
    );
    Ok(chosen)
}

fn create_device(
    instance: &Instance,
    physical: &PhysicalDevice,
) -> anyhow::Result<(ash::Device, u32, vk::Queue)> {
    let queue_families = unsafe {
        instance
            .handle()
            .get_physical_device_queue_family_properties(physical.handle())
    };
    let queue_family_index = queue_families
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .or_else(|| queue_families.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::TRANSFER)))
        .context("no suitable queue family")? as u32;

    let required_exts: Vec<&'static CStr> = vec![
        ext::image_drm_format_modifier::NAME,
        ext::external_memory_dma_buf::NAME,
        khr::external_memory_fd::NAME,
        khr::external_semaphore_fd::NAME,
        khr::external_semaphore::NAME,
    ];
    for e in &required_exts {
        if !physical.has_device_extension(*e) {
            anyhow::bail!("required device extension missing: {:?}", e);
        }
    }
    let mut exts = required_exts;
    // Optional but useful for correct ownership transfers with non-Vulkan producers (EGL/GL/XWayland).
    if physical.has_device_extension(ext::queue_family_foreign::NAME) {
        exts.push(ext::queue_family_foreign::NAME);
    }
    let ext_ptrs = exts.iter().map(|e| e.as_ptr()).collect::<Vec<_>>();

    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&[1.0])];
    let create = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_info)
        .enabled_extension_names(&ext_ptrs);

    let device = unsafe {
        instance
            .handle()
            .create_device(physical.handle(), &create, None)
    }
    .context("create_device")?;
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
    Ok((device, queue_family_index, queue))
}

fn query_modifiers(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    format: vk::Format,
) -> anyhow::Result<Vec<u64>> {
    let force_linear = std::env::var("METABONK_DMABUF_ONLY_LINEAR")
        .ok()
        .map(|v| v.trim() == "1")
        .unwrap_or(false)
        || std::env::var("METABONK_FORCE_LINEAR_EXPORT")
            .ok()
            .map(|v| v.trim() == "1")
            .unwrap_or(false);

    let mut list0 = vk::DrmFormatModifierPropertiesListEXT::default();
    let mut props0 = vk::FormatProperties2::default().push_next(&mut list0);
    unsafe { instance.get_physical_device_format_properties2(physical, format, &mut props0) };
    let count = list0.drm_format_modifier_count;
    if count == 0 {
        anyhow::bail!("no DRM modifiers reported for format {format:?}");
    }
    let mut vec_props = vec![vk::DrmFormatModifierPropertiesEXT::default(); count as usize];
    let mut list = vk::DrmFormatModifierPropertiesListEXT::default().drm_format_modifier_properties(&mut vec_props);
    let mut props = vk::FormatProperties2::default().push_next(&mut list);
    unsafe { instance.get_physical_device_format_properties2(physical, format, &mut props) };

    let mut out: Vec<u64> = Vec::new();
    if force_linear {
        let linear = u64::from(DrmModifier::Linear);
        for p in &vec_props {
            if p.drm_format_modifier == linear {
                out.push(p.drm_format_modifier);
            }
        }
        if !out.is_empty() {
            debug!(
                "format {format:?} forcing linear modifier only ({} candidates)",
                out.len()
            );
            return Ok(out);
        }
        warn!("format {format:?} no linear modifier available; falling back to full list");
    }
    for p in &vec_props {
        // Prefer formats usable for transfer dst (we just clear via transfer).
        if p.drm_format_modifier_tiling_features.contains(vk::FormatFeatureFlags::TRANSFER_DST) {
            out.push(p.drm_format_modifier);
        }
    }
    if out.is_empty() {
        for p in &vec_props {
            out.push(p.drm_format_modifier);
        }
    }
    // If linear exists, keep it as a fallback but prefer driver-optimal first.
    out.sort_by_key(|m| if *m == u64::from(DrmModifier::Linear) { 1 } else { 0 });
    debug!("format {format:?} modifiers: {} candidates", out.len());
    Ok(out)
}

fn find_memory_type(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    req: vk::MemoryRequirements,
    device_local: bool,
) -> anyhow::Result<u32> {
    let props = unsafe { instance.get_physical_device_memory_properties(physical) };
    for i in 0..props.memory_type_count {
        let ok = (req.memory_type_bits & (1 << i)) != 0;
        if !ok {
            continue;
        }
        let flags = props.memory_types[i as usize].property_flags;
        if device_local && flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            return Ok(i);
        }
        if !device_local {
            return Ok(i);
        }
    }
    anyhow::bail!("no suitable memory type found")
}

fn find_memory_type_bits(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    allowed_type_bits: u32,
    device_local: bool,
) -> anyhow::Result<u32> {
    let props = unsafe { instance.get_physical_device_memory_properties(physical) };
    for i in 0..props.memory_type_count {
        let ok = (allowed_type_bits & (1 << i)) != 0;
        if !ok {
            continue;
        }
        let flags = props.memory_types[i as usize].property_flags;
        if device_local && flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            return Ok(i);
        }
        if !device_local {
            return Ok(i);
        }
    }
    anyhow::bail!("no suitable memory type found for allowed bits 0x{allowed_type_bits:x}")
}

fn find_memory_type_with_flags(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    allowed_type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> anyhow::Result<u32> {
    let props = unsafe { instance.get_physical_device_memory_properties(physical) };
    for i in 0..props.memory_type_count {
        let ok = (allowed_type_bits & (1 << i)) != 0;
        if !ok {
            continue;
        }
        let flags = props.memory_types[i as usize].property_flags;
        if flags.contains(required) {
            return Ok(i);
        }
    }
    anyhow::bail!(
        "no suitable memory type found for allowed bits 0x{allowed_type_bits:x} with flags {required:?}"
    )
}

fn select_best_memory_type_bits(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    allowed_type_bits: u32,
) -> anyhow::Result<u32> {
    let props = unsafe { instance.get_physical_device_memory_properties(physical) };
    let mut best: Option<(u32, u64, bool)> = None;
    for i in 0..props.memory_type_count {
        let ok = (allowed_type_bits & (1 << i)) != 0;
        if !ok {
            continue;
        }
        let mt = props.memory_types[i as usize];
        let flags = mt.property_flags;
        let heap_idx = mt.heap_index as usize;
        let heap_size = props.memory_heaps.get(heap_idx).map(|h| h.size).unwrap_or(0);
        let is_device_local = flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
        match best {
            None => best = Some((i, heap_size, is_device_local)),
            Some((_bi, bsize, bdev)) => {
                // Prefer DEVICE_LOCAL, then prefer larger heap size.
                if (is_device_local && !bdev) || (is_device_local == bdev && heap_size > bsize) {
                    best = Some((i, heap_size, is_device_local));
                }
            }
        }
    }
    if let Some((idx, _sz, _dev)) = best {
        return Ok(idx);
    }
    anyhow::bail!("no suitable memory type found for allowed bits 0x{allowed_type_bits:x}")
}

fn describe_memory_type_bits(instance: &ash::Instance, physical: vk::PhysicalDevice, allowed_type_bits: u32) -> String {
    let props = unsafe { instance.get_physical_device_memory_properties(physical) };
    let mut out = Vec::new();
    for i in 0..props.memory_type_count {
        if (allowed_type_bits & (1 << i)) == 0 {
            continue;
        }
        let mt = props.memory_types[i as usize];
        let flags = mt.property_flags;
        let heap_idx = mt.heap_index as usize;
        let heap_size = props.memory_heaps.get(heap_idx).map(|h| h.size).unwrap_or(0);
        out.push(format!(
            "t{} heap={}MiB flags={:?}",
            i,
            heap_size / (1024 * 1024),
            flags
        ));
    }
    if out.is_empty() {
        return "no_allowed_types".to_string();
    }
    format!("types=[{}]", out.join(" "))
}

fn dmabuf_key(src: &Dmabuf) -> u64 {
    let mut hasher = DefaultHasher::new();
    src.format().code.hash(&mut hasher);
    let modifier_u64: u64 = src.format().modifier.into();
    modifier_u64.hash(&mut hasher);
    src.width().hash(&mut hasher);
    src.height().hash(&mut hasher);
    src.strides().next().unwrap_or(0).hash(&mut hasher);
    src.offsets().next().unwrap_or(0).hash(&mut hasher);
    if let Some(h) = src.handles().next() {
        let fd = h.as_raw_fd();
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut st) } == 0 {
            (st.st_ino as u64).hash(&mut hasher);
        }
    }
    hasher.finish()
}
