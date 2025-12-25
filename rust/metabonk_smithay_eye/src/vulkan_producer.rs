use std::{
    ffi::CStr,
    os::fd::AsRawFd,
    os::unix::io::RawFd,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use anyhow::Context;
use ash::{
    ext,
    khr,
    vk,
};
use drm_fourcc::{DrmFourcc, DrmModifier};
use nix::unistd::close;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::{Buffer as _, Fourcc as ShmFourcc};
use smithay::backend::vulkan::{version::Version, AppInfo, Instance, PhysicalDevice};
use tracing::{debug, info, warn};

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
}

#[derive(Clone, Debug)]
struct SrcCache {
    key: u64,
    image: vk::Image,
    memory: vk::DeviceMemory,
    width: u32,
    height: u32,
    layout: vk::ImageLayout,
}

struct Slot {
    image: vk::Image,
    memory: vk::DeviceMemory,
    modifier: u64,
    stride: u32,
    offset: u32,
    mem_size: u64,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    prev_acquire: Option<vk::Semaphore>,
    prev_release: Option<vk::Semaphore>,
    to_destroy: Vec<(vk::Semaphore, vk::Semaphore)>,
    to_destroy_imported: Vec<(vk::Image, vk::DeviceMemory)>,
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

        let mut producer = Self {
            instance,
            physical,
            device,
            queue,
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
        };
        producer.init_slots(slots)?;
        Ok(producer)
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
                cmd: cmds[idx],
                fence,
                prev_acquire: None,
                prev_release: None,
                to_destroy: Vec::new(),
                to_destroy_imported: Vec::new(),
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
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
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

    pub fn render_next(&mut self, clear_phase: u8) -> anyhow::Result<RenderedFrame> {
        let slot_idx = self.next_slot % self.slots.len();
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        {
            let fence = self.slots[slot_idx].fence;
            // Wait for previous GPU work on this slot (allows safe semaphore destruction/reset).
            self.wait_for_fence(fence).context("wait_for_fences")?;
            let slot = &mut self.slots[slot_idx];
            unsafe { self.device.reset_fences(&[slot.fence]) }.ok();
            for (a, r) in slot.to_destroy.drain(..) {
                unsafe {
                    self.device.destroy_semaphore(a, None);
                    self.device.destroy_semaphore(r, None);
                }
            }
            for (img, mem) in slot.to_destroy_imported.drain(..) {
                unsafe {
                    self.device.destroy_image(img, None);
                    self.device.free_memory(mem, None);
                }
            }
        }

        let (acquire, acquire_fd) = self.create_exportable_semaphore()?;
        let (release, release_fd) = self.create_exportable_semaphore()?;

        // Record: clear image to a varying color (keeps MVP visually obvious for debugging).
        let (image, cmd, fence, prev_acquire, prev_release, modifier, stride, offset, mem_size, memory) = {
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
            )
        };

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
            self.device.end_command_buffer(cmd)?;
        }

        // Submit with explicit wait on previous consumer release.
        let mut wait_sems: Vec<vk::Semaphore> = Vec::new();
        let mut wait_stages: Vec<vk::PipelineStageFlags> = Vec::new();
        if let Some(prev_rel) = prev_release {
            wait_sems.push(prev_rel);
            wait_stages.push(vk::PipelineStageFlags::TRANSFER);
        }
        let signal_sems = [acquire];

        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&cmd))
            .signal_semaphores(&signal_sems);

        unsafe { self.device.queue_submit(self.queue, &[submit], fence) }.context("queue_submit")?;

        // Export DMABuf FD for the memory backing this image.
        let dmabuf_fd = self.export_memory_dmabuf_fd(memory)?;

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
            modifier,
            stride,
            offset,
            size_bytes: mem_size as u32,
            dmabuf_fd,
            acquire_fence_fd: acquire_fd,
            release_fence_fd: release_fd,
        })
    }

    pub fn render_from_dmabuf(&mut self, src: &Dmabuf) -> anyhow::Result<RenderedFrame> {
        let slot_idx = self.next_slot % self.slots.len();
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        {
            let fence = self.slots[slot_idx].fence;
            self.wait_for_fence(fence).context("wait_for_fences")?;
            let slot = &mut self.slots[slot_idx];
            unsafe { self.device.reset_fences(&[slot.fence]) }.ok();
            for (a, r) in slot.to_destroy.drain(..) {
                unsafe {
                    self.device.destroy_semaphore(a, None);
                    self.device.destroy_semaphore(r, None);
                }
            }
            for (img, mem) in slot.to_destroy_imported.drain(..) {
                unsafe {
                    self.device.destroy_image(img, None);
                    self.device.free_memory(mem, None);
                }
            }
        }

        let (acquire, acquire_fd) = self.create_exportable_semaphore()?;
        let (release, release_fd) = self.create_exportable_semaphore()?;

        // Import the source DMA-BUF as a Vulkan image for GPU-to-GPU copy into our exportable slot image.
        //
        // Critical: do NOT import/allocate a fresh Vulkan image for every frame. Even if the producer
        // is reusing a small set of DMA-BUFs (triple buffering), importing every tick can exhaust
        // driver resources and lead to ERROR_OUT_OF_DEVICE_MEMORY. Reuse the last import until the
        // DMA-BUF identity changes.
        let src_key = dmabuf_key(src);
        let mut src_layout = vk::ImageLayout::UNDEFINED;
        let mut imported_new = false;
        let (src_image, src_mem, src_w, src_h) = if let Some(c) = self.last_src.as_ref() {
            if c.key == src_key {
                src_layout = c.layout;
                (c.image, c.memory, c.width, c.height)
            } else {
                // Defer destruction of the previous import until the destination fence signals.
                if let Some(old) = self.last_src.take() {
                    let slot = &mut self.slots[slot_idx];
                    slot.to_destroy_imported.push((old.image, old.memory));
                }
                let (img, mem, w, h) = self.import_dmabuf_image(src)?;
                imported_new = true;
                self.last_src = Some(SrcCache {
                    key: src_key,
                    image: img,
                    memory: mem,
                    width: w,
                    height: h,
                    layout: vk::ImageLayout::UNDEFINED,
                });
                (img, mem, w, h)
            }
        } else {
            let (img, mem, w, h) = self.import_dmabuf_image(src)?;
            imported_new = true;
            self.last_src = Some(SrcCache {
                key: src_key,
                image: img,
                memory: mem,
                width: w,
                height: h,
                layout: vk::ImageLayout::UNDEFINED,
            });
            (img, mem, w, h)
        };

        let (dst_image, cmd, fence, prev_acquire, prev_release, modifier, stride, offset, mem_size, memory) = {
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
            )
        };

        if src_w != self.width || src_h != self.height {
            // For now, require a matching input size. (This avoids an extra blit path and keeps the ABI stable.)
            // We still send a RESET so the worker drops the segment rather than training on a frozen eye.
            if imported_new {
                // Not used by any GPU submission yet; safe to destroy immediately.
                unsafe {
                    self.device.destroy_image(src_image, None);
                    self.device.free_memory(src_mem, None);
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

            let src_to_src = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(src_layout)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
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
            self.device.cmd_copy_image(
                cmd,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
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
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
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

            self.device.end_command_buffer(cmd)?;
        }
        if let Some(c) = self.last_src.as_mut() {
            if c.key == src_key {
                c.layout = vk::ImageLayout::GENERAL;
            }
        }

        // Submit with explicit wait on previous consumer release.
        let mut wait_sems: Vec<vk::Semaphore> = Vec::new();
        let mut wait_stages: Vec<vk::PipelineStageFlags> = Vec::new();
        if let Some(prev_rel) = prev_release {
            wait_sems.push(prev_rel);
            wait_stages.push(vk::PipelineStageFlags::TRANSFER);
        }
        let signal_sems = [acquire];

        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&cmd))
            .signal_semaphores(&signal_sems);

        unsafe { self.device.queue_submit(self.queue, &[submit], fence) }.context("queue_submit")?;

        // Export DMABuf FD for the memory backing this image.
        let dmabuf_fd = self.export_memory_dmabuf_fd(memory)?;

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
            modifier,
            stride,
            offset,
            size_bytes: mem_size as u32,
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
    ) -> anyhow::Result<RenderedFrame> {
        if src.num_planes() != 1 {
            anyhow::bail!("only single-plane dmabufs are supported (got {} planes)", src.num_planes());
        }

        let slot_idx = self.next_slot % self.slots.len();
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        {
            let fence = self.slots[slot_idx].fence;
            self.wait_for_fence(fence).context("wait_for_fences")?;
            let slot = &mut self.slots[slot_idx];
            unsafe { self.device.reset_fences(&[slot.fence]) }.ok();
            for (a, r) in slot.to_destroy.drain(..) {
                unsafe {
                    self.device.destroy_semaphore(a, None);
                    self.device.destroy_semaphore(r, None);
                }
            }
            for (img, mem) in slot.to_destroy_imported.drain(..) {
                unsafe {
                    self.device.destroy_image(img, None);
                    self.device.free_memory(mem, None);
                }
            }
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

        let (acquire, acquire_fd) = self.create_exportable_semaphore()?;
        let (release, release_fd) = self.create_exportable_semaphore()?;

        // Submit a trivial no-op command buffer that waits on previous consumer release (backpressure)
        // and signals acquire. We intentionally submit a real command buffer (even empty) because
        // some driver stacks appear to produce external semaphore handles that CUDA can import
        // reliably only after being used in a queue submission with at least one command buffer.
        // Passthrough mode does not strictly depend on consumer release fences. Waiting on them
        // can deadlock the compositor if the consumer misses a single fence signal (e.g. CUDA import hiccup).
        // Keep the producer running and rely on the fixed-FPS pacing + slot fences for backpressure.
        let wait_sems: Vec<vk::Semaphore> = Vec::new();
        let wait_stages: Vec<vk::PipelineStageFlags> = Vec::new();

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

    fn import_dmabuf_image(&self, src: &Dmabuf) -> anyhow::Result<(vk::Image, vk::DeviceMemory, u32, u32)> {
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

        // Some drivers require a non-zero `size` for explicit modifier images.
        // For our single-plane ARGB path, approximate the plane size as stride * height.
        let plane_layout = vk::SubresourceLayout {
            offset,
            size: stride.saturating_mul(height as u64),
            row_pitch: stride,
            array_pitch: 0,
            depth_pitch: 0,
        };
        let src_fd = src.handles().next().unwrap();
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
                .initial_layout(vk::ImageLayout::UNDEFINED);
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

        Ok((image, memory, width, height))
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
                self.device.destroy_image(c.image, None);
                self.device.free_memory(c.memory, None);
            }
        }
    }

    fn wait_for_fence(&self, fence: vk::Fence) -> anyhow::Result<()> {
        // Avoid indefinite hangs. If the GPU queue wedges (e.g. external semaphore never signals),
        // we prefer a fast failure so the supervisor can restart the compositor.
        // 30s gives the CUDA consumer time to catch up under load without spurious restarts.
        const WAIT_NS: u64 = 30_000_000_000; // 30s
        match unsafe { self.device.wait_for_fences(&[fence], true, WAIT_NS) } {
            Ok(_) => Ok(()),
            Err(vk::Result::TIMEOUT) => anyhow::bail!("slot fence wait timed out (GPU queue stalled)"),
            Err(e) => Err(anyhow::anyhow!("slot fence wait failed: {e:?}")),
        }
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
}

impl Drop for VulkanProducer {
    fn drop(&mut self) {
        unsafe {
            if let Some(c) = self.last_src.take() {
                self.device.destroy_image(c.image, None);
                self.device.free_memory(c.memory, None);
            }
            for slot in self.slots.drain(..) {
                for (img, mem) in slot.to_destroy_imported {
                    self.device.destroy_image(img, None);
                    self.device.free_memory(mem, None);
                }
                for (a, r) in slot.to_destroy {
                    self.device.destroy_semaphore(a, None);
                    self.device.destroy_semaphore(r, None);
                }
                if let Some(a) = slot.prev_acquire {
                    self.device.destroy_semaphore(a, None);
                }
                if let Some(r) = slot.prev_release {
                    self.device.destroy_semaphore(r, None);
                }
                self.device.destroy_fence(slot.fence, None);
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

    let exts: Vec<&'static CStr> = vec![
        ext::image_drm_format_modifier::NAME,
        ext::external_memory_dma_buf::NAME,
        khr::external_memory_fd::NAME,
        khr::external_semaphore_fd::NAME,
        khr::external_semaphore::NAME,
    ];
    for e in &exts {
        if !physical.has_device_extension(*e) {
            anyhow::bail!("required device extension missing: {:?}", e);
        }
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
