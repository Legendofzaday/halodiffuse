use rayon::prelude::*;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::env::var;
use std::fs::read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
    PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::sync::GpuFuture;
use vulkano::{DeviceSize, Version, VulkanLibrary};

#[repr(C)]
#[derive(Clone, Copy, BufferContents)]
pub struct EulerPushConstants {
    pub dt: f32,
}

#[repr(C)]
#[derive(Clone, Copy, BufferContents)]
pub struct NoisePushConstants {
    pub seed: u64,
    pub num_pairs: u32,
    pub _pad: u32,
}

macro_rules! include_spv {
    ($stem:literal) => {{
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/", $stem, ".spv"));
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<u32>>()
    }};
}

pub fn get_data_dir() -> PathBuf {
    PathBuf::from(var("HOME").unwrap()).join(".local/share/halodiffuse")
}

pub struct WeightPack {
    tensors: HashMap<String, Subbuffer<[u8]>>,
}

impl WeightPack {
    pub fn load(engine: &Engine, staging: &Subbuffer<[u8]>, path: impl AsRef<Path>) -> Self {
        let raw_file = read(path.as_ref()).unwrap();
        let st = SafeTensors::deserialize(&raw_file).unwrap();
        let mut items: Vec<(String, &[u8])> = st
            .iter()
            .map(|(name, view)| (name.to_string(), view.data()))
            .collect();
        items.sort_by_key(|(_, data)| data.as_ptr() as usize);
        let mut tensors = HashMap::with_capacity(items.len());
        let mut destinations = Vec::with_capacity(items.len());
        for (name, data) in &items {
            let dst = engine.alloc_device_weight(data.len() as DeviceSize);
            tensors.insert(name.clone(), dst.clone());
            destinations.push(dst);
        }
        let staging_cap = staging.size() as usize;
        let mut idx = 0;
        while idx < items.len() {
            let (_, first_data) = items[idx];
            if first_data.len() > staging_cap {
                let mut offset = 0usize;
                while offset < first_data.len() {
                    let chunk_len = (first_data.len() - offset).min(staging_cap);
                    {
                        let mut guard = staging.write().unwrap();
                        guard[..chunk_len].copy_from_slice(&first_data[offset..offset + chunk_len]);
                    }
                    let mut cmd = engine.new_one_time_cmd();
                    cmd.copy_buffer(CopyBufferInfo::buffers(
                        staging.clone().slice(0..(chunk_len as DeviceSize)),
                        destinations[idx]
                            .clone()
                            .slice((offset as DeviceSize)..((offset + chunk_len) as DeviceSize)),
                    ))
                    .unwrap();
                    engine.submit_and_wait(cmd);
                    offset += chunk_len;
                }
                idx += 1;
                continue;
            }
            let batch_start = idx;
            let mut stage_cursor = 0usize;
            let mut batch_layout = Vec::new();
            while idx < items.len() {
                let len = items[idx].1.len();
                let padded = (len + 15) & !15;
                if stage_cursor + padded > staging_cap {
                    break;
                }
                batch_layout.push((stage_cursor, len, padded));
                stage_cursor += padded;
                idx += 1;
            }
            {
                let mut guard = staging.write().unwrap();
                let mut rest: &mut [u8] = &mut guard[..stage_cursor];
                let mut dst_slices = Vec::with_capacity(batch_layout.len());
                for &(_, len, padded) in &batch_layout {
                    let (head, tail) = rest.split_at_mut(padded);
                    dst_slices.push(&mut head[..len]);
                    rest = tail;
                }
                dst_slices.par_iter_mut().enumerate().for_each(|(i, dst)| {
                    dst.copy_from_slice(items[batch_start + i].1);
                });
            }
            let mut cmd = engine.new_one_time_cmd();
            for (i, &(src_off, len, _)) in batch_layout.iter().enumerate() {
                if len == 0 {
                    continue;
                }
                cmd.copy_buffer(CopyBufferInfo::buffers(
                    staging
                        .clone()
                        .slice((src_off as DeviceSize)..((src_off + len) as DeviceSize)),
                    destinations[batch_start + i]
                        .clone()
                        .slice(0..(len as DeviceSize)),
                ))
                .unwrap();
            }
            engine.submit_and_wait(cmd);
        }
        Self { tensors }
    }

    pub fn get(&self, key: &str) -> Subbuffer<[u8]> {
        self.tensors
            .get(key)
            .unwrap_or_else(|| panic!("missing weight tensor: {key}"))
            .clone()
    }
}

pub struct DeviceScratchArena {
    pub arena_a: Subbuffer<[u8]>,
    pub arena_b: Subbuffer<[u8]>,
    pub arena_c: Subbuffer<[u8]>,
    pub align: DeviceSize,
}

impl DeviceScratchArena {
    pub fn new(engine: &Engine) -> Self {
        let max_vae_bytes: DeviceSize = 1024 * 1024 * 256 * 2;
        Self {
            arena_a: engine.alloc_device_bytes(max_vae_bytes),
            arena_b: engine.alloc_device_bytes(max_vae_bytes),
            arena_c: engine.alloc_device_bytes(max_vae_bytes),
            align: engine.storage_offset_align,
        }
    }
}

pub struct ArenaBump<'a> {
    buf: &'a Subbuffer<[u8]>,
    offset: DeviceSize,
    align: DeviceSize,
}

impl<'a> ArenaBump<'a> {
    pub fn new(buf: &'a Subbuffer<[u8]>, align: DeviceSize) -> Self {
        Self {
            buf,
            offset: 0,
            align,
        }
    }

    pub fn alloc(&mut self, bytes: DeviceSize) -> Subbuffer<[u8]> {
        let start = (self.offset + self.align - 1) & !(self.align - 1);
        let end = start + bytes;
        assert!(
            end <= self.buf.size(),
            "DeviceScratchArena overflow: requested {end} > {}",
            self.buf.size()
        );
        self.offset = end;
        self.buf.clone().slice(start..end)
    }
}

pub struct BoundDispatch {
    pipeline: Arc<ComputePipeline>,
    layout: Arc<PipelineLayout>,
    set: Arc<DescriptorSet>,
    grid: [u32; 3],
}

impl BoundDispatch {
    pub fn record(&self, cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        cmd.bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.layout.clone(),
                0,
                self.set.clone(),
            )
            .unwrap();
        unsafe {
            cmd.dispatch(self.grid).unwrap();
        }
    }

    pub fn record_push<P: BufferContents>(
        &self,
        cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        push: P,
    ) {
        cmd.bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.layout.clone(),
                0,
                self.set.clone(),
            )
            .unwrap()
            .push_constants(self.layout.clone(), 0, push)
            .unwrap();
        unsafe {
            cmd.dispatch(self.grid).unwrap();
        }
    }
}

pub struct Engine {
    device: Arc<Device>,
    queue: Arc<Queue>,
    mem_alloc: Arc<StandardMemoryAllocator>,
    desc_alloc: Arc<StandardDescriptorSetAllocator>,
    cmd_alloc: Arc<StandardCommandBufferAllocator>,
    pipelines: HashMap<&'static str, Arc<ComputePipeline>>,
    storage_offset_align: DeviceSize,
}

impl Engine {
    pub fn new() -> Self {
        let library = VulkanLibrary::new().unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                max_api_version: Some(Version::V1_3),
                ..Default::default()
            },
        )
        .unwrap();
        let physical_device = instance
            .enumerate_physical_devices()
            .unwrap()
            .find(|p| {
                p.properties().vendor_id == 0x1002
                    && p.properties().device_type
                        == vulkano::device::physical::PhysicalDeviceType::IntegratedGpu
            })
            .unwrap();
        let storage_offset_align = u64::from(
            physical_device
                .properties()
                .min_storage_buffer_offset_alignment,
        )
        .max(256);
        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .position(|q| {
                q.queue_flags.intersects(QueueFlags::COMPUTE)
                    && !q.queue_flags.intersects(QueueFlags::GRAPHICS)
            })
            .unwrap() as u32;
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: DeviceExtensions {
                    khr_cooperative_matrix: true,
                    khr_shader_float16_int8: true,
                    khr_16bit_storage: true,
                    khr_8bit_storage: true,
                    ..DeviceExtensions::empty()
                },
                enabled_features: DeviceFeatures {
                    vulkan_memory_model: true,
                    vulkan_memory_model_device_scope: true,
                    shader_float16: true,
                    shader_int8: true,
                    shader_int64: true,
                    storage_buffer16_bit_access: true,
                    uniform_and_storage_buffer16_bit_access: true,
                    storage_buffer8_bit_access: true,
                    uniform_and_storage_buffer8_bit_access: true,
                    subgroup_size_control: true,
                    compute_full_subgroups: true,
                    shader_subgroup_extended_types: true,
                    cooperative_matrix: true,
                    maintenance4: true,
                    synchronization2: true,
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let queue = queues.next().unwrap();
        let mem_alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let desc_alloc = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let cmd_alloc = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let mut engine = Self {
            device,
            queue,
            mem_alloc,
            desc_alloc,
            cmd_alloc,
            pipelines: HashMap::new(),
            storage_offset_align,
        };
        engine.compile_all_pipelines();
        engine
    }

    fn add_pipeline(&mut self, name: &'static str, words: &[u32]) {
        let module =
            unsafe { ShaderModule::new(self.device.clone(), ShaderModuleCreateInfo::new(words)) }
                .unwrap();
        let entry_point = module.single_entry_point().unwrap();
        let mut stage = PipelineShaderStageCreateInfo::new(entry_point);
        stage.required_subgroup_size = Some(32);
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(self.device.clone())
                .unwrap(),
        )
        .unwrap();
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap();
        self.pipelines.insert(name, pipeline);
    }

    fn compile_all_pipelines(&mut self) {
        macro_rules! load {
            ($($stem:literal),* $(,)?) => {
                $( self.add_pipeline(concat!($stem, ".comp"), &include_spv!($stem)); )*
            };
        }
        load!(
            "conv2d_1x1_c256_f128_h1024_w1024_bias_residual",
            "conv2d_1x1_c512_f1536_h128_w128_bias",
            "conv2d_1x1_c512_f256_h512_w512_bias_residual",
            "conv2d_1x1_c512_f512_h128_w128_bias_residual",
            "conv2d_3x3_c128_f128_h1024_w1024_bias",
            "conv2d_3x3_c128_f128_h1024_w1024_bias_residual",
            "conv2d_3x3_c128_f3_h1024_w1024_bias_clamp_rgb8",
            "conv2d_3x3_c16_f512_h128_w128_bias",
            "conv2d_3x3_c256_f128_h1024_w1024_bias",
            "conv2d_3x3_c256_f256_h512_w512_bias",
            "conv2d_3x3_c256_f256_h512_w512_bias_residual",
            "conv2d_3x3_c512_f256_h512_w512_bias",
            "conv2d_3x3_c512_f512_h128_w128_bias",
            "conv2d_3x3_c512_f512_h128_w128_bias_residual",
            "conv2d_3x3_c512_f512_h256_w256_bias",
            "conv2d_3x3_c512_f512_h256_w256_bias_residual",
            "conv2d_3x3_upsample2x_c256_f256_h1024_w1024_bias",
            "conv2d_3x3_upsample2x_c512_f512_h256_w256_bias",
            "conv2d_3x3_upsample2x_c512_f512_h512_w512_bias",
            "flash_attention_m16384_h1_d512",
            "flow_match_euler_step_m4096_k64",
            "gaussian_noise_splitmix64",
            "groupnorm_g32_c128_h1024_w1024_silu",
            "groupnorm_g32_c256_h1024_w1024_silu",
            "groupnorm_g32_c256_h512_w512_silu",
            "groupnorm_g32_c512_h128_w128",
            "groupnorm_g32_c512_h128_w128_silu",
            "groupnorm_g32_c512_h256_w256_silu",
            "groupnorm_g32_c512_h512_w512_silu",
            "adaln_mod_m1_n14592_k2432_bias",
            "adaln_mod_m1_n4864_k2432_bias",
            "clip_concat_pad_m77_d4096",
            "embed_lookup_pos_v49408_m77_d1280",
            "embed_lookup_pos_v49408_m77_d768",
            "embed_lookup_v32128_m256_d4096",
            "flash_attention_causal_m77_h12_d64",
            "flash_attention_causal_m77_h20_d64",
            "flash_attention_rpb_m256_h64_d64",
            "flash_attention_v_concat_m4429_h38_d64",
            "hgemm_m1_n2432_k2048_bias_silu",
            "hgemm_m1_n2432_k2432_bias",
            "hgemm_m1_n2432_k2432_bias_cond_add_silu",
            "hgemm_m256_n10240_k4096_geglu",
            "hgemm_m256_n12288_k4096",
            "hgemm_m256_n4096_k10240_residual",
            "hgemm_m256_n4096_k4096_residual",
            "hgemm_m333_n2432_k2432_bias_gated_residual",
            "hgemm_m333_n2432_k4096_bias",
            "hgemm_m333_n2432_k9728_bias_gated_residual",
            "hgemm_m333_n7296_k2432_bias",
            "hgemm_m333_n9728_k2432_bias_gelu",
            "hgemm_m4096_n2432_k2432_bias_gated_residual",
            "hgemm_m4096_n2432_k9728_bias_gated_residual",
            "hgemm_m4096_n64_k2432_bias",
            "hgemm_m4096_n7296_k2432_bias",
            "hgemm_m4096_n9728_k2432_bias_gelu",
            "hgemm_m77_n1280_k1280_bias_residual",
            "hgemm_m77_n1280_k5120_bias_residual",
            "hgemm_m77_n2304_k768_bias",
            "hgemm_m77_n3072_k768_bias_gelu",
            "hgemm_m77_n3840_k1280_bias",
            "hgemm_m77_n5120_k1280_bias_gelu",
            "hgemm_m77_n768_k3072_bias_residual",
            "hgemm_m77_n768_k768_bias_residual",
            "layernorm_adaln_m333_d2432",
            "layernorm_adaln_m4096_d2432",
            "layernorm_eos_hgemm_m1_n1280_k1280",
            "layernorm_eos_hgemm_m1_n768_k768",
            "layernorm_m77_d1280",
            "layernorm_m77_d768",
            "patch_embed_conv2d_m4096_n2432_k64",
            "qk_rmsnorm_concat_h38_d64",
            "rmsnorm_m256_d4096",
            "timestep_embed_sinusoidal_hgemm_m1_n2432_k256_bias_silu",
            "unpatchify_m4096_k64_to_latents_c16_h128_w128_euler_scale",
            "adaln_mod_m1_n15360_k256_bias_tanh",
            "adaln_mod_m1_n3840_k256_silu_bias",
            "embed_lookup_v151936_m256_d2560",
            "flash_attention_causal_gqa_m256_h32_hkv8_d128",
            "flash_attention_m256_h30_d128",
            "flash_attention_m4096_h30_d128",
            "flash_attention_m4352_h30_d128",
            "hgemm_m1_n256_k1024_bias",
            "hgemm_m256_4096_4352_n10240_k3840_swiglu",
            "hgemm_m256_4096_4352_n11520_k3840",
            "hgemm_m256_4096_4352_n3840_k10240",
            "hgemm_m256_4096_4352_n3840_k3840",
            "hgemm_m256_n2560_k4096_residual",
            "hgemm_m256_n2560_k9728_residual",
            "hgemm_m256_n3840_k2560_bias_cap_pad",
            "hgemm_m256_n6144_k2560",
            "hgemm_m256_n9728_k2560_swiglu",
            "hgemm_m4096_n64_k3840_bias",
            "layernorm_adaln_scale_m4096_d3840",
            "patch_embed_conv2d_m4096_n3840_k64",
            "qk_rmsnorm_rope1d_m256_h32_hkv8_d128",
            "qk_rmsnorm_rope3d_m256_h30_d128",
            "qk_rmsnorm_rope3d_m4096_h30_d128",
            "qk_rmsnorm_rope3d_m4352_h30_d128",
            "rmsnorm_adaln_m4096_4352_d3840",
            "rmsnorm_d3840_gated_residual_m4096_4352",
            "rmsnorm_m256_d2560",
            "rmsnorm_m256_d3840",
            "rmsnorm_m256_d3840_residual",
            "timestep_embed_sinusoidal_hgemm_m1_n1024_k256_bias_silu",
            "unpatchify_m4096_k64_to_latents_c16_h128_w128_euler_scale_flux",
        );
    }

    pub fn alloc_device_weight(&self, size: DeviceSize) -> Subbuffer<[u8]> {
        Buffer::new_slice::<u8>(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            size.max(16),
        )
        .unwrap()
    }

    pub fn alloc_device_bytes(&self, size: DeviceSize) -> Subbuffer<[u8]> {
        Buffer::new_slice::<u8>(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            size,
        )
        .unwrap()
    }

    pub fn alloc_staging_bytes(&self, size: DeviceSize) -> Subbuffer<[u8]> {
        Buffer::new_slice::<u8>(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            size,
        )
        .unwrap()
    }

    pub fn alloc_host_bytes(&self, size: DeviceSize) -> Subbuffer<[u8]> {
        Buffer::new_slice::<u8>(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            size,
        )
        .unwrap()
    }

    pub fn bind(
        &self,
        shader: &'static str,
        grid: [u32; 3],
        buffers: &[Subbuffer<[u8]>],
    ) -> BoundDispatch {
        let pipeline = self.pipelines.get(shader).unwrap().clone();
        let layout = pipeline.layout().clone();
        let set_layout = layout.set_layouts()[0].clone();
        let writes: Vec<WriteDescriptorSet> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| WriteDescriptorSet::buffer(i as u32, buf.clone()))
            .collect();
        let set = DescriptorSet::new(self.desc_alloc.clone(), set_layout, writes, []).unwrap();
        BoundDispatch {
            pipeline,
            layout,
            set,
            grid,
        }
    }

    pub fn new_one_time_cmd(&self) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.cmd_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
    }

    pub fn new_reusable_cmd(&self) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.cmd_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap()
    }

    pub fn submit_and_wait(&self, cmd: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        cmd.build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    pub fn submit_recorded_and_wait(&self, cmd: &Arc<PrimaryAutoCommandBuffer>) {
        cmd.clone()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }
}
