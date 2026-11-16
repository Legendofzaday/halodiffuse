#![feature(f16)]

mod install;

use rayon::prelude::*;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::env::{args, current_dir, current_exe, var};
use std::fs::{File, create_dir_all, read, read_to_string, remove_file, write};
use std::io::{BufRead, BufReader, BufWriter, ErrorKind, Read, Write, stdout};
use std::mem::swap;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, exit, id};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::sleep;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
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
struct EulerPushConstants {
    dt: f32,
}

#[repr(C)]
#[derive(Clone, Copy, BufferContents)]
struct NoisePushConstants {
    seed: u64,
    num_pairs: u32,
    _pad: u32,
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

struct WeightPack {
    tensors: HashMap<String, Subbuffer<[u8]>>,
}

impl WeightPack {
    fn load(engine: &Engine, staging: &Subbuffer<[u8]>, path: impl AsRef<Path>) -> Self {
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

    fn get(&self, key: &str) -> Subbuffer<[u8]> {
        self.tensors
            .get(key)
            .unwrap_or_else(|| panic!("missing weight tensor: {key}"))
            .clone()
    }
}

struct DeviceScratchArena {
    arena_a: Subbuffer<[u8]>,
    arena_b: Subbuffer<[u8]>,
    arena_c: Subbuffer<[u8]>,
    align: DeviceSize,
}

impl DeviceScratchArena {
    fn new(engine: &Engine) -> Self {
        let max_vae_bytes: DeviceSize = 1024 * 1024 * 256 * 2;
        Self {
            arena_a: engine.alloc_device_bytes(max_vae_bytes),
            arena_b: engine.alloc_device_bytes(max_vae_bytes),
            arena_c: engine.alloc_device_bytes(max_vae_bytes),
            align: engine.storage_offset_align,
        }
    }
}

struct ArenaBump<'a> {
    buf: &'a Subbuffer<[u8]>,
    offset: DeviceSize,
    align: DeviceSize,
}

impl<'a> ArenaBump<'a> {
    fn new(buf: &'a Subbuffer<[u8]>, align: DeviceSize) -> Self {
        Self {
            buf,
            offset: 0,
            align,
        }
    }

    fn alloc(&mut self, bytes: DeviceSize) -> Subbuffer<[u8]> {
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

struct ClipScratch {
    x: Subbuffer<[u8]>,
    norm: Subbuffer<[u8]>,
    qkv: Subbuffer<[u8]>,
    attn_out: Subbuffer<[u8]>,
    mlp_hid: Subbuffer<[u8]>,
}

struct TextEncoderWorkspace {
    clip_l: ClipScratch,
    clip_g: ClipScratch,
    t5_x: Subbuffer<[u8]>,
    t5_norm: Subbuffer<[u8]>,
    t5_qkv: Subbuffer<[u8]>,
    t5_attn_out: Subbuffer<[u8]>,
    t5_ffn_hid: Subbuffer<[u8]>,
}

impl TextEncoderWorkspace {
    fn new(arena: &DeviceScratchArena) -> Self {
        let mut bump = ArenaBump::new(&arena.arena_a, arena.align);
        let mut alloc_clip = |d: DeviceSize| -> ClipScratch {
            let bytes_seq = 77 * d * 2;
            ClipScratch {
                x: bump.alloc(bytes_seq),
                norm: bump.alloc(bytes_seq),
                qkv: bump.alloc(bytes_seq * 3),
                attn_out: bump.alloc(bytes_seq),
                mlp_hid: bump.alloc(bytes_seq * 4),
            }
        };
        let clip_l = alloc_clip(768);
        let clip_g = alloc_clip(1280);
        let t5_seq = 256 * 4096 * 2;
        Self {
            clip_l,
            clip_g,
            t5_x: bump.alloc(t5_seq),
            t5_norm: bump.alloc(t5_seq),
            t5_qkv: bump.alloc(t5_seq * 3),
            t5_attn_out: bump.alloc(t5_seq),
            t5_ffn_hid: bump.alloc(256 * 10240 * 2),
        }
    }
}

struct MmditWorkspace {
    t_silu: Subbuffer<[u8]>,
    t_out: Subbuffer<[u8]>,
    y_silu: Subbuffer<[u8]>,
    c_silu: Subbuffer<[u8]>,
    x: Subbuffer<[u8]>,
    ctx: Subbuffer<[u8]>,
    adaln_dual: Subbuffer<[u8]>,
    adaln_2: Subbuffer<[u8]>,
    x_norm: Subbuffer<[u8]>,
    ctx_norm: Subbuffer<[u8]>,
    x_qkv: Subbuffer<[u8]>,
    ctx_qkv: Subbuffer<[u8]>,
    qk_concat: Subbuffer<[u8]>,
    attn_concat: Subbuffer<[u8]>,
    x_mlp: Subbuffer<[u8]>,
    ctx_mlp: Subbuffer<[u8]>,
}

impl MmditWorkspace {
    fn new(arena: &DeviceScratchArena) -> Self {
        let mut bump = ArenaBump::new(&arena.arena_a, arena.align);
        let d_bytes = 2432 * 2;
        Self {
            t_silu: bump.alloc(d_bytes),
            t_out: bump.alloc(d_bytes),
            y_silu: bump.alloc(d_bytes),
            c_silu: bump.alloc(d_bytes),
            x: bump.alloc(4096 * d_bytes),
            ctx: bump.alloc(333 * d_bytes),
            adaln_dual: bump.alloc(12 * d_bytes),
            adaln_2: bump.alloc(2 * d_bytes),
            x_norm: bump.alloc(4096 * d_bytes),
            ctx_norm: bump.alloc(333 * d_bytes),
            x_qkv: bump.alloc(4096 * 3 * d_bytes),
            ctx_qkv: bump.alloc(333 * 3 * d_bytes),
            qk_concat: bump.alloc(4429 * 2 * d_bytes),
            attn_concat: bump.alloc(4429 * d_bytes),
            x_mlp: bump.alloc(4096 * 4 * d_bytes),
            ctx_mlp: bump.alloc(333 * 4 * d_bytes),
        }
    }
}

struct VaeWorkspace {
    x: Subbuffer<[u8]>,
    norm: Subbuffer<[u8]>,
    tmp: Subbuffer<[u8]>,
    qkv_128: Subbuffer<[u8]>,
}

impl VaeWorkspace {
    fn new(engine: &Engine, arena: &DeviceScratchArena) -> Self {
        Self {
            x: arena.arena_a.clone(),
            norm: arena.arena_b.clone(),
            tmp: arena.arena_c.clone(),
            qkv_128: engine.alloc_device_bytes(128 * 128 * 1536 * 2),
        }
    }
}

struct BoundDispatch {
    pipeline: Arc<ComputePipeline>,
    layout: Arc<PipelineLayout>,
    set: Arc<DescriptorSet>,
    grid: [u32; 3],
}

impl BoundDispatch {
    fn record(&self, cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
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

    fn record_push<P: BufferContents>(
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

struct PrebuiltClip {
    emb: BoundDispatch,
    layers: Vec<[BoundDispatch; 7]>,
    copy_penultimate: (Subbuffer<[u8]>, Subbuffer<[u8]>),
    eos_dispatches: Vec<BoundDispatch>,
}

struct PrebuiltTextEncoders {
    noise: BoundDispatch,
    clip_l: PrebuiltClip,
    clip_g: PrebuiltClip,
    clip_concat: BoundDispatch,
    t5_emb: BoundDispatch,
    t5_layers: Vec<[BoundDispatch; 7]>,
    t5_final_ln: BoundDispatch,
}

struct Engine {
    device: Arc<Device>,
    queue: Arc<Queue>,
    mem_alloc: Arc<StandardMemoryAllocator>,
    desc_alloc: Arc<StandardDescriptorSetAllocator>,
    cmd_alloc: Arc<StandardCommandBufferAllocator>,
    pipelines: HashMap<&'static str, Arc<ComputePipeline>>,
    storage_offset_align: DeviceSize,
}

impl Engine {
    fn new() -> Self {
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
            .next()
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
            .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
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
            "adaln_mod_m1_n14592_k2432_bias",
            "adaln_mod_m1_n4864_k2432_bias",
            "clip_concat_pad_m77_d4096",
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
            "embed_lookup_pos_v49408_m77_d1280",
            "embed_lookup_pos_v49408_m77_d768",
            "embed_lookup_v32128_m256_d4096",
            "flash_attention_causal_m77_h12_d64",
            "flash_attention_causal_m77_h20_d64",
            "flash_attention_m16384_h1_d512",
            "flash_attention_rpb_m256_h64_d64",
            "flash_attention_v_concat_m4429_h38_d64",
            "flow_match_euler_step_m4096_k64",
            "gaussian_noise_splitmix64",
            "groupnorm_g32_c128_h1024_w1024_silu",
            "groupnorm_g32_c256_h1024_w1024_silu",
            "groupnorm_g32_c256_h512_w512_silu",
            "groupnorm_g32_c512_h128_w128",
            "groupnorm_g32_c512_h128_w128_silu",
            "groupnorm_g32_c512_h256_w256_silu",
            "groupnorm_g32_c512_h512_w512_silu",
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
        );
    }

    fn alloc_device_weight(&self, size: DeviceSize) -> Subbuffer<[u8]> {
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

    fn alloc_device_bytes(&self, size: DeviceSize) -> Subbuffer<[u8]> {
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

    fn alloc_staging_bytes(&self, size: DeviceSize) -> Subbuffer<[u8]> {
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

    fn alloc_host_bytes(&self, size: DeviceSize) -> Subbuffer<[u8]> {
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

    fn bind(
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

    fn new_one_time_cmd(&self) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.cmd_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
    }

    fn new_reusable_cmd(&self) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.cmd_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap()
    }

    fn submit_and_wait(&self, cmd: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        cmd.build()
            .unwrap()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    fn submit_recorded_and_wait(&self, cmd: &Arc<PrimaryAutoCommandBuffer>) {
        cmd.clone()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    fn build_clip(
        &self,
        w: &WeightPack,
        scratch: &ClipScratch,
        tokens_buf: &Subbuffer<[u8]>,
        d: usize,
        num_layers: usize,
        out_seq: &Subbuffer<[u8]>,
        out_pooled: &Subbuffer<[u8]>,
    ) -> PrebuiltClip {
        let (sh_emb, sh_ln, sh_qkv, sh_fa, sh_out, sh_fc1, sh_fc2, sh_eos) = if d == 1280 {
            (
                "embed_lookup_pos_v49408_m77_d1280.comp",
                "layernorm_m77_d1280.comp",
                "hgemm_m77_n3840_k1280_bias.comp",
                "flash_attention_causal_m77_h20_d64.comp",
                "hgemm_m77_n1280_k1280_bias_residual.comp",
                "hgemm_m77_n5120_k1280_bias_gelu.comp",
                "hgemm_m77_n1280_k5120_bias_residual.comp",
                "layernorm_eos_hgemm_m1_n1280_k1280.comp",
            )
        } else {
            (
                "embed_lookup_pos_v49408_m77_d768.comp",
                "layernorm_m77_d768.comp",
                "hgemm_m77_n2304_k768_bias.comp",
                "flash_attention_causal_m77_h12_d64.comp",
                "hgemm_m77_n768_k768_bias_residual.comp",
                "hgemm_m77_n3072_k768_bias_gelu.comp",
                "hgemm_m77_n768_k3072_bias_residual.comp",
                "layernorm_eos_hgemm_m1_n768_k768.comp",
            )
        };
        let nx_d = (d / 128) as u32;
        let heads = (d / 64) as u32;
        let emb = self.bind(
            sh_emb,
            [77, 1, 1],
            &[
                tokens_buf.clone(),
                w.get("embeddings.token_embedding.weight"),
                w.get("embeddings.position_embedding.weight"),
                scratch.x.clone(),
            ],
        );
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push([
                self.bind(
                    sh_ln,
                    [77, 1, 1],
                    &[
                        scratch.x.clone(),
                        w.get(&format!("layers.{i}.ln1.gamma")),
                        w.get(&format!("layers.{i}.ln1.beta")),
                        scratch.norm.clone(),
                    ],
                ),
                self.bind(
                    sh_qkv,
                    [nx_d * 3, 3, 1],
                    &[
                        scratch.norm.clone(),
                        w.get(&format!("layers.{i}.attn.qkv.weight")),
                        w.get(&format!("layers.{i}.attn.qkv.bias")),
                        scratch.qkv.clone(),
                    ],
                ),
                self.bind(
                    sh_fa,
                    [1, heads, 1],
                    &[scratch.qkv.clone(), scratch.attn_out.clone()],
                ),
                self.bind(
                    sh_out,
                    [nx_d, 3, 1],
                    &[
                        scratch.attn_out.clone(),
                        w.get(&format!("layers.{i}.attn.out_proj.weight")),
                        w.get(&format!("layers.{i}.attn.out_proj.bias")),
                        scratch.x.clone(),
                    ],
                ),
                self.bind(
                    sh_ln,
                    [77, 1, 1],
                    &[
                        scratch.x.clone(),
                        w.get(&format!("layers.{i}.ln2.gamma")),
                        w.get(&format!("layers.{i}.ln2.beta")),
                        scratch.norm.clone(),
                    ],
                ),
                self.bind(
                    sh_fc1,
                    [nx_d * 4, 3, 1],
                    &[
                        scratch.norm.clone(),
                        w.get(&format!("layers.{i}.mlp.fc1.weight")),
                        w.get(&format!("layers.{i}.mlp.fc1.bias")),
                        scratch.mlp_hid.clone(),
                    ],
                ),
                self.bind(
                    sh_fc2,
                    [nx_d, 3, 1],
                    &[
                        scratch.mlp_hid.clone(),
                        w.get(&format!("layers.{i}.mlp.fc2.weight")),
                        w.get(&format!("layers.{i}.mlp.fc2.bias")),
                        scratch.x.clone(),
                    ],
                ),
            ]);
        }
        let row_bytes = (d * 2) as DeviceSize;
        let final_gamma = w.get("final_ln.gamma");
        let final_beta = w.get("final_ln.beta");
        let eos_weight = w.get("eos_proj.weight");
        let eos_bias = w.get("eos_proj.bias");
        let eos_dispatches = (0..77u64)
            .map(|eos_idx| {
                let eos_slice = scratch
                    .x
                    .clone()
                    .slice((eos_idx * row_bytes)..((eos_idx + 1) * row_bytes));
                self.bind(
                    sh_eos,
                    [nx_d, 1, 1],
                    &[
                        eos_slice,
                        final_gamma.clone(),
                        final_beta.clone(),
                        eos_weight.clone(),
                        eos_bias.clone(),
                        out_pooled.clone(),
                    ],
                )
            })
            .collect();
        PrebuiltClip {
            emb,
            layers,
            copy_penultimate: (scratch.x.clone(), out_seq.clone()),
            eos_dispatches,
        }
    }

    fn build_text_encoders(
        &self,
        ws: &TextEncoderWorkspace,
        w_clip_l: &WeightPack,
        w_clip_g: &WeightPack,
        w_t5: &WeightPack,
        latents_x: &Subbuffer<[u8]>,
        buf_tokens_l: &Subbuffer<[u8]>,
        buf_tokens_g: &Subbuffer<[u8]>,
        buf_tokens_t5: &Subbuffer<[u8]>,
        clip_l_seq: &Subbuffer<[u8]>,
        clip_g_seq: &Subbuffer<[u8]>,
        y_pooled: &Subbuffer<[u8]>,
        ctx_333: &Subbuffer<[u8]>,
    ) -> PrebuiltTextEncoders {
        let num_pairs = (latents_x.size() / 4) as u32;
        let noise = self.bind(
            "gaussian_noise_splitmix64.comp",
            [num_pairs.div_ceil(128), 1, 1],
            &[latents_x.clone()],
        );
        let y_l_slice = y_pooled.clone().slice(0..(768 * 2));
        let y_g_slice = y_pooled.clone().slice((768 * 2)..(2048 * 2));
        let clip_l = self.build_clip(
            w_clip_l,
            &ws.clip_l,
            buf_tokens_l,
            768,
            12,
            clip_l_seq,
            &y_l_slice,
        );
        let clip_g = self.build_clip(
            w_clip_g,
            &ws.clip_g,
            buf_tokens_g,
            1280,
            32,
            clip_g_seq,
            &y_g_slice,
        );
        let ctx_clip_slice = ctx_333.clone().slice(0..(77 * 4096 * 2));
        let clip_concat = self.bind(
            "clip_concat_pad_m77_d4096.comp",
            [77, 1, 1],
            &[clip_l_seq.clone(), clip_g_seq.clone(), ctx_clip_slice],
        );
        let rpb = w_t5.get("rpb_weight");
        let t5_emb = self.bind(
            "embed_lookup_v32128_m256_d4096.comp",
            [256, 1, 1],
            &[
                buf_tokens_t5.clone(),
                w_t5.get("embed_tokens.weight"),
                ws.t5_x.clone(),
            ],
        );
        let mut t5_layers = Vec::with_capacity(24);
        for i in 0..24 {
            t5_layers.push([
                self.bind(
                    "rmsnorm_m256_d4096.comp",
                    [256, 1, 1],
                    &[
                        ws.t5_x.clone(),
                        w_t5.get(&format!("blocks.{i}.ln1.gamma")),
                        ws.t5_norm.clone(),
                    ],
                ),
                self.bind(
                    "hgemm_m256_n12288_k4096.comp",
                    [96, 2, 1],
                    &[
                        ws.t5_norm.clone(),
                        w_t5.get(&format!("blocks.{i}.attn.qkv.weight")),
                        ws.t5_qkv.clone(),
                    ],
                ),
                self.bind(
                    "flash_attention_rpb_m256_h64_d64.comp",
                    [4, 64, 1],
                    &[ws.t5_qkv.clone(), rpb.clone(), ws.t5_attn_out.clone()],
                ),
                self.bind(
                    "hgemm_m256_n4096_k4096_residual.comp",
                    [32, 2, 1],
                    &[
                        ws.t5_attn_out.clone(),
                        w_t5.get(&format!("blocks.{i}.attn.o.weight")),
                        ws.t5_x.clone(),
                    ],
                ),
                self.bind(
                    "rmsnorm_m256_d4096.comp",
                    [256, 1, 1],
                    &[
                        ws.t5_x.clone(),
                        w_t5.get(&format!("blocks.{i}.ln2.gamma")),
                        ws.t5_norm.clone(),
                    ],
                ),
                self.bind(
                    "hgemm_m256_n10240_k4096_geglu.comp",
                    [160, 2, 1],
                    &[
                        ws.t5_norm.clone(),
                        w_t5.get(&format!("blocks.{i}.ffn.b0_wi0.weight")),
                        w_t5.get(&format!("blocks.{i}.ffn.b1_wi1.weight")),
                        ws.t5_ffn_hid.clone(),
                    ],
                ),
                self.bind(
                    "hgemm_m256_n4096_k10240_residual.comp",
                    [32, 2, 1],
                    &[
                        ws.t5_ffn_hid.clone(),
                        w_t5.get(&format!("blocks.{i}.ffn.wo.weight")),
                        ws.t5_x.clone(),
                    ],
                ),
            ]);
        }
        let ctx_t5_slice = ctx_333.clone().slice((77 * 4096 * 2)..(333 * 4096 * 2));
        let t5_final_ln = self.bind(
            "rmsnorm_m256_d4096.comp",
            [256, 1, 1],
            &[ws.t5_x.clone(), w_t5.get("final_ln.gamma"), ctx_t5_slice],
        );
        PrebuiltTextEncoders {
            noise,
            clip_l,
            clip_g,
            clip_concat,
            t5_emb,
            t5_layers,
            t5_final_ln,
        }
    }

    fn run_stage1_text_and_noise(
        &self,
        pre: &PrebuiltTextEncoders,
        seed: u64,
        num_pairs: u32,
        eos_l: u64,
        eos_g: u64,
    ) {
        let mut cmd = self.new_one_time_cmd();
        pre.noise.record_push(
            &mut cmd,
            NoisePushConstants {
                seed,
                num_pairs,
                _pad: 0,
            },
        );
        let record_clip = |cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                           clip: &PrebuiltClip,
                           eos_idx: u64| {
            clip.emb.record(cmd);
            let penultimate = clip.layers.len() - 2;
            for (i, layer) in clip.layers.iter().enumerate() {
                for disp in layer {
                    disp.record(cmd);
                }
                if i == penultimate {
                    cmd.copy_buffer(CopyBufferInfo::buffers(
                        clip.copy_penultimate.0.clone(),
                        clip.copy_penultimate.1.clone(),
                    ))
                    .unwrap();
                }
            }
            clip.eos_dispatches[eos_idx as usize].record(cmd);
        };
        record_clip(&mut cmd, &pre.clip_l, eos_l);
        record_clip(&mut cmd, &pre.clip_g, eos_g);
        pre.clip_concat.record(&mut cmd);
        pre.t5_emb.record(&mut cmd);
        for layer in &pre.t5_layers {
            for disp in layer {
                disp.record(&mut cmd);
            }
        }
        pre.t5_final_ln.record(&mut cmd);
        self.submit_and_wait(cmd);
    }

    fn record_mmdit_step_cmd(
        &self,
        ws: &MmditWorkspace,
        w: &WeightPack,
        latents_x: &Subbuffer<[u8]>,
        ctx_in: &Subbuffer<[u8]>,
        y_pooled: &Subbuffer<[u8]>,
        t_buf: &Subbuffer<[u8]>,
        v_out: &Subbuffer<[u8]>,
        euler_out_nchw: Option<&Subbuffer<[u8]>>,
        dt: f32,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let d_bytes = 2432 * 2;
        let chunk = |buf: &Subbuffer<[u8]>, idx: u64| -> Subbuffer<[u8]> {
            buf.clone().slice((idx * d_bytes)..((idx + 1) * d_bytes))
        };
        let mut cmd = self.new_reusable_cmd();
        self.bind(
            "timestep_embed_sinusoidal_hgemm_m1_n2432_k256_bias_silu.comp",
            [19, 1, 1],
            &[
                t_buf.clone(),
                w.get("t_embedder.mlp.0.weight"),
                w.get("t_embedder.mlp.0.bias"),
                ws.t_silu.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "hgemm_m1_n2432_k2432_bias.comp",
            [19, 1, 1],
            &[
                ws.t_silu.clone(),
                w.get("t_embedder.mlp.2.weight"),
                w.get("t_embedder.mlp.2.bias"),
                ws.t_out.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "hgemm_m1_n2432_k2048_bias_silu.comp",
            [19, 1, 1],
            &[
                y_pooled.clone(),
                w.get("y_embedder.mlp.0.weight"),
                w.get("y_embedder.mlp.0.bias"),
                ws.y_silu.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "hgemm_m1_n2432_k2432_bias_cond_add_silu.comp",
            [19, 1, 1],
            &[
                ws.y_silu.clone(),
                w.get("y_embedder.mlp.2.weight"),
                w.get("y_embedder.mlp.2.bias"),
                ws.t_out.clone(),
                ws.c_silu.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "patch_embed_conv2d_m4096_n2432_k64.comp",
            [19, 32, 1],
            &[
                latents_x.clone(),
                w.get("x_embedder.proj.weight"),
                w.get("x_embedder.fused_pos_bias"),
                ws.x.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "hgemm_m333_n2432_k4096_bias.comp",
            [19, 11, 1],
            &[
                ctx_in.clone(),
                w.get("context_embedder.weight"),
                w.get("context_embedder.bias"),
                ws.ctx.clone(),
            ],
        )
        .record(&mut cmd);
        for i in 0..38 {
            let is_last = i == 37;
            if !is_last {
                self.bind(
                    "adaln_mod_m1_n14592_k2432_bias.comp",
                    [114, 2, 1],
                    &[
                        ws.c_silu.clone(),
                        w.get(&format!("blocks.{i}.adaln_14592.weight")),
                        w.get(&format!("blocks.{i}.adaln_14592.bias")),
                        ws.adaln_dual.clone(),
                    ],
                )
                .record(&mut cmd);
            } else {
                self.bind(
                    "adaln_mod_m1_n14592_k2432_bias.comp",
                    [114, 1, 1],
                    &[
                        ws.c_silu.clone(),
                        w.get(&format!("blocks.{i}.x_adaln_14592.weight")),
                        w.get(&format!("blocks.{i}.x_adaln_14592.bias")),
                        ws.adaln_dual.clone(),
                    ],
                )
                .record(&mut cmd);
                self.bind(
                    "adaln_mod_m1_n4864_k2432_bias.comp",
                    [38, 1, 1],
                    &[
                        ws.c_silu.clone(),
                        w.get(&format!("blocks.{i}.ctx_adaln_4864.weight")),
                        w.get(&format!("blocks.{i}.ctx_adaln_4864.bias")),
                        ws.adaln_2.clone(),
                    ],
                )
                .record(&mut cmd);
            }
            self.bind(
                "layernorm_adaln_m4096_d2432.comp",
                [4096, 1, 1],
                &[
                    ws.x.clone(),
                    chunk(&ws.adaln_dual, 1),
                    chunk(&ws.adaln_dual, 0),
                    ws.x_norm.clone(),
                ],
            )
            .record(&mut cmd);
            let (ctx_gamma_msa, ctx_beta_msa) = if !is_last {
                (chunk(&ws.adaln_dual, 7), chunk(&ws.adaln_dual, 6))
            } else {
                (chunk(&ws.adaln_2, 1), chunk(&ws.adaln_2, 0))
            };
            self.bind(
                "layernorm_adaln_m333_d2432.comp",
                [333, 1, 1],
                &[
                    ws.ctx.clone(),
                    ctx_gamma_msa,
                    ctx_beta_msa,
                    ws.ctx_norm.clone(),
                ],
            )
            .record(&mut cmd);
            self.bind(
                "hgemm_m4096_n7296_k2432_bias.comp",
                [57, 32, 1],
                &[
                    ws.x_norm.clone(),
                    w.get(&format!("blocks.{i}.x_attn.qkv.weight")),
                    w.get(&format!("blocks.{i}.x_attn.qkv.bias")),
                    ws.x_qkv.clone(),
                ],
            )
            .record(&mut cmd);
            self.bind(
                "hgemm_m333_n7296_k2432_bias.comp",
                [57, 11, 1],
                &[
                    ws.ctx_norm.clone(),
                    w.get(&format!("blocks.{i}.ctx_attn.qkv.weight")),
                    w.get(&format!("blocks.{i}.ctx_attn.qkv.bias")),
                    ws.ctx_qkv.clone(),
                ],
            )
            .record(&mut cmd);
            self.bind(
                "qk_rmsnorm_concat_h38_d64.comp",
                [4429, 1, 1],
                &[
                    ws.x_qkv.clone(),
                    ws.ctx_qkv.clone(),
                    w.get(&format!("blocks.{i}.qk_norm.gamma_x_q")),
                    w.get(&format!("blocks.{i}.qk_norm.gamma_x_k")),
                    w.get(&format!("blocks.{i}.qk_norm.gamma_ctx_q")),
                    w.get(&format!("blocks.{i}.qk_norm.gamma_ctx_k")),
                    ws.qk_concat.clone(),
                ],
            )
            .record(&mut cmd);
            self.bind(
                "flash_attention_v_concat_m4429_h38_d64.comp",
                [70, 38, 1],
                &[
                    ws.qk_concat.clone(),
                    ws.x_qkv.clone(),
                    ws.ctx_qkv.clone(),
                    ws.attn_concat.clone(),
                ],
            )
            .record(&mut cmd);
            let x_attn_slice = ws.attn_concat.clone().slice(0..(4096 * d_bytes));
            let ctx_attn_slice = ws
                .attn_concat
                .clone()
                .slice((4096 * d_bytes)..(4429 * d_bytes));
            self.bind(
                "hgemm_m4096_n2432_k2432_bias_gated_residual.comp",
                [19, 32, 1],
                &[
                    x_attn_slice,
                    w.get(&format!("blocks.{i}.x_attn.proj.weight")),
                    w.get(&format!("blocks.{i}.x_attn.proj.bias")),
                    chunk(&ws.adaln_dual, 2),
                    ws.x.clone(),
                ],
            )
            .record(&mut cmd);
            if !is_last {
                self.bind(
                    "hgemm_m333_n2432_k2432_bias_gated_residual.comp",
                    [19, 11, 1],
                    &[
                        ctx_attn_slice,
                        w.get(&format!("blocks.{i}.ctx_attn.proj.weight")),
                        w.get(&format!("blocks.{i}.ctx_attn.proj.bias")),
                        chunk(&ws.adaln_dual, 8),
                        ws.ctx.clone(),
                    ],
                )
                .record(&mut cmd);
            }
            self.bind(
                "layernorm_adaln_m4096_d2432.comp",
                [4096, 1, 1],
                &[
                    ws.x.clone(),
                    chunk(&ws.adaln_dual, 4),
                    chunk(&ws.adaln_dual, 3),
                    ws.x_norm.clone(),
                ],
            )
            .record(&mut cmd);
            if !is_last {
                self.bind(
                    "layernorm_adaln_m333_d2432.comp",
                    [333, 1, 1],
                    &[
                        ws.ctx.clone(),
                        chunk(&ws.adaln_dual, 10),
                        chunk(&ws.adaln_dual, 9),
                        ws.ctx_norm.clone(),
                    ],
                )
                .record(&mut cmd);
            }
            self.bind(
                "hgemm_m4096_n9728_k2432_bias_gelu.comp",
                [76, 32, 1],
                &[
                    ws.x_norm.clone(),
                    w.get(&format!("blocks.{i}.x_mlp.fc1.weight")),
                    w.get(&format!("blocks.{i}.x_mlp.fc1.bias")),
                    ws.x_mlp.clone(),
                ],
            )
            .record(&mut cmd);
            if !is_last {
                self.bind(
                    "hgemm_m333_n9728_k2432_bias_gelu.comp",
                    [76, 11, 1],
                    &[
                        ws.ctx_norm.clone(),
                        w.get(&format!("blocks.{i}.ctx_mlp.fc1.weight")),
                        w.get(&format!("blocks.{i}.ctx_mlp.fc1.bias")),
                        ws.ctx_mlp.clone(),
                    ],
                )
                .record(&mut cmd);
            }
            self.bind(
                "hgemm_m4096_n2432_k9728_bias_gated_residual.comp",
                [19, 32, 1],
                &[
                    ws.x_mlp.clone(),
                    w.get(&format!("blocks.{i}.x_mlp.fc2.weight")),
                    w.get(&format!("blocks.{i}.x_mlp.fc2.bias")),
                    chunk(&ws.adaln_dual, 5),
                    ws.x.clone(),
                ],
            )
            .record(&mut cmd);
            if !is_last {
                self.bind(
                    "hgemm_m333_n2432_k9728_bias_gated_residual.comp",
                    [19, 11, 1],
                    &[
                        ws.ctx_mlp.clone(),
                        w.get(&format!("blocks.{i}.ctx_mlp.fc2.weight")),
                        w.get(&format!("blocks.{i}.ctx_mlp.fc2.bias")),
                        chunk(&ws.adaln_dual, 11),
                        ws.ctx.clone(),
                    ],
                )
                .record(&mut cmd);
            }
        }
        self.bind(
            "adaln_mod_m1_n4864_k2432_bias.comp",
            [38, 1, 1],
            &[
                ws.c_silu.clone(),
                w.get("final_layer.adaln_4864.weight"),
                w.get("final_layer.adaln_4864.bias"),
                ws.adaln_2.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "layernorm_adaln_m4096_d2432.comp",
            [4096, 1, 1],
            &[
                ws.x.clone(),
                chunk(&ws.adaln_2, 1),
                chunk(&ws.adaln_2, 0),
                ws.x_norm.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "hgemm_m4096_n64_k2432_bias.comp",
            [32, 1, 1],
            &[
                ws.x_norm.clone(),
                w.get("final_layer.linear.weight"),
                w.get("final_layer.linear.bias"),
                v_out.clone(),
            ],
        )
        .record(&mut cmd);
        if let Some(nchw_out) = euler_out_nchw {
            self.bind(
                "unpatchify_m4096_k64_to_latents_c16_h128_w128_euler_scale.comp",
                [256, 1, 1],
                &[v_out.clone(), latents_x.clone(), nchw_out.clone()],
            )
            .record_push(&mut cmd, EulerPushConstants { dt });
        } else {
            self.bind(
                "flow_match_euler_step_m4096_k64.comp",
                [1024, 1, 1],
                &[v_out.clone(), latents_x.clone()],
            )
            .record_push(&mut cmd, EulerPushConstants { dt });
        }
        cmd.build().unwrap()
    }

    fn record_vae_cmd(
        &self,
        ws: &VaeWorkspace,
        w: &WeightPack,
        latents_nchw: &Subbuffer<[u8]>,
        rgb_out: &Subbuffer<[u8]>,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let mut x = ws.x.clone();
        let norm = ws.norm.clone();
        let mut tmp = ws.tmp.clone();
        let qkv_128 = ws.qkv_128.clone();
        let resnet_same = |cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                           p: &str,
                           gn_sh: &'static str,
                           c1_sh: &'static str,
                           c2_res_sh: &'static str,
                           c1_grid: [u32; 3],
                           c2_grid: [u32; 3],
                           x_buf: &Subbuffer<[u8]>,
                           n_buf: &Subbuffer<[u8]>,
                           t_buf: &Subbuffer<[u8]>| {
            self.bind(
                gn_sh,
                [32, 1, 1],
                &[
                    x_buf.clone(),
                    w.get(&format!("{p}.norm1.gamma")),
                    w.get(&format!("{p}.norm1.beta")),
                    n_buf.clone(),
                ],
            )
            .record(cmd);
            self.bind(
                c1_sh,
                c1_grid,
                &[
                    n_buf.clone(),
                    w.get(&format!("{p}.conv1.weight")),
                    w.get(&format!("{p}.conv1.bias")),
                    t_buf.clone(),
                ],
            )
            .record(cmd);
            self.bind(
                gn_sh,
                [32, 1, 1],
                &[
                    t_buf.clone(),
                    w.get(&format!("{p}.norm2.gamma")),
                    w.get(&format!("{p}.norm2.beta")),
                    n_buf.clone(),
                ],
            )
            .record(cmd);
            self.bind(
                c2_res_sh,
                c2_grid,
                &[
                    n_buf.clone(),
                    w.get(&format!("{p}.conv2.weight")),
                    w.get(&format!("{p}.conv2.bias")),
                    x_buf.clone(),
                ],
            )
            .record(cmd);
        };
        let mut cmd = self.new_reusable_cmd();
        self.bind(
            "conv2d_3x3_c16_f512_h128_w128_bias.comp",
            [4, 128, 1],
            &[
                latents_nchw.clone(),
                w.get("conv_in.weight"),
                w.get("conv_in.bias"),
                x.clone(),
            ],
        )
        .record(&mut cmd);
        resnet_same(
            &mut cmd,
            "mid.block_1",
            "groupnorm_g32_c512_h128_w128_silu.comp",
            "conv2d_3x3_c512_f512_h128_w128_bias.comp",
            "conv2d_3x3_c512_f512_h128_w128_bias_residual.comp",
            [4, 128, 1],
            [4, 128, 1],
            &x,
            &norm,
            &tmp,
        );
        self.bind(
            "groupnorm_g32_c512_h128_w128.comp",
            [32, 1, 1],
            &[
                x.clone(),
                w.get("mid.attn.norm.gamma"),
                w.get("mid.attn.norm.beta"),
                norm.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "conv2d_1x1_c512_f1536_h128_w128_bias.comp",
            [12, 128, 1],
            &[
                norm.clone(),
                w.get("mid.attn.qkv.weight"),
                w.get("mid.attn.qkv.bias"),
                qkv_128.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "flash_attention_m16384_h1_d512.comp",
            [1024, 1, 1],
            &[qkv_128, tmp.clone()],
        )
        .record(&mut cmd);
        self.bind(
            "conv2d_1x1_c512_f512_h128_w128_bias_residual.comp",
            [4, 128, 1],
            &[
                tmp.clone(),
                w.get("mid.attn.proj_out.weight"),
                w.get("mid.attn.proj_out.bias"),
                x.clone(),
            ],
        )
        .record(&mut cmd);
        resnet_same(
            &mut cmd,
            "mid.block_2",
            "groupnorm_g32_c512_h128_w128_silu.comp",
            "conv2d_3x3_c512_f512_h128_w128_bias.comp",
            "conv2d_3x3_c512_f512_h128_w128_bias_residual.comp",
            [4, 128, 1],
            [4, 128, 1],
            &x,
            &norm,
            &tmp,
        );
        for r in 0..3 {
            resnet_same(
                &mut cmd,
                &format!("up.0.res.{r}"),
                "groupnorm_g32_c512_h128_w128_silu.comp",
                "conv2d_3x3_c512_f512_h128_w128_bias.comp",
                "conv2d_3x3_c512_f512_h128_w128_bias_residual.comp",
                [4, 128, 1],
                [4, 128, 1],
                &x,
                &norm,
                &tmp,
            );
        }
        self.bind(
            "conv2d_3x3_upsample2x_c512_f512_h256_w256_bias.comp",
            [4, 512, 1],
            &[
                x.clone(),
                w.get("up.0.upsample_conv.weight"),
                w.get("up.0.upsample_conv.bias"),
                tmp.clone(),
            ],
        )
        .record(&mut cmd);
        swap(&mut x, &mut tmp);
        for r in 0..3 {
            resnet_same(
                &mut cmd,
                &format!("up.1.res.{r}"),
                "groupnorm_g32_c512_h256_w256_silu.comp",
                "conv2d_3x3_c512_f512_h256_w256_bias.comp",
                "conv2d_3x3_c512_f512_h256_w256_bias_residual.comp",
                [4, 512, 1],
                [4, 512, 1],
                &x,
                &norm,
                &tmp,
            );
        }
        self.bind(
            "conv2d_3x3_upsample2x_c512_f512_h512_w512_bias.comp",
            [4, 2048, 1],
            &[
                x.clone(),
                w.get("up.1.upsample_conv.weight"),
                w.get("up.1.upsample_conv.bias"),
                tmp.clone(),
            ],
        )
        .record(&mut cmd);
        swap(&mut x, &mut tmp);
        self.bind(
            "groupnorm_g32_c512_h512_w512_silu.comp",
            [32, 1, 1],
            &[
                x.clone(),
                w.get("up.2.res.0.norm1.gamma"),
                w.get("up.2.res.0.norm1.beta"),
                norm.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "conv2d_3x3_c512_f256_h512_w512_bias.comp",
            [2, 2048, 1],
            &[
                norm.clone(),
                w.get("up.2.res.0.conv1.weight"),
                w.get("up.2.res.0.conv1.bias"),
                tmp.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "groupnorm_g32_c256_h512_w512_silu.comp",
            [32, 1, 1],
            &[
                tmp.clone(),
                w.get("up.2.res.0.norm2.gamma"),
                w.get("up.2.res.0.norm2.beta"),
                norm.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "conv2d_3x3_c256_f256_h512_w512_bias.comp",
            [2, 2048, 1],
            &[
                norm.clone(),
                w.get("up.2.res.0.conv2.weight"),
                w.get("up.2.res.0.conv2.bias"),
                tmp.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "conv2d_1x1_c512_f256_h512_w512_bias_residual.comp",
            [2, 2048, 1],
            &[
                x.clone(),
                w.get("up.2.res.0.conv_shortcut.weight"),
                w.get("up.2.res.0.conv_shortcut.bias"),
                tmp.clone(),
            ],
        )
        .record(&mut cmd);
        swap(&mut x, &mut tmp);
        for r in 1..3 {
            resnet_same(
                &mut cmd,
                &format!("up.2.res.{r}"),
                "groupnorm_g32_c256_h512_w512_silu.comp",
                "conv2d_3x3_c256_f256_h512_w512_bias.comp",
                "conv2d_3x3_c256_f256_h512_w512_bias_residual.comp",
                [2, 2048, 1],
                [2, 2048, 1],
                &x,
                &norm,
                &tmp,
            );
        }
        self.bind(
            "conv2d_3x3_upsample2x_c256_f256_h1024_w1024_bias.comp",
            [2, 8192, 1],
            &[
                x.clone(),
                w.get("up.2.upsample_conv.weight"),
                w.get("up.2.upsample_conv.bias"),
                tmp.clone(),
            ],
        )
        .record(&mut cmd);
        swap(&mut x, &mut tmp);
        self.bind(
            "groupnorm_g32_c256_h1024_w1024_silu.comp",
            [32, 1, 1],
            &[
                x.clone(),
                w.get("up.3.res.0.norm1.gamma"),
                w.get("up.3.res.0.norm1.beta"),
                norm.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "conv2d_3x3_c256_f128_h1024_w1024_bias.comp",
            [16384, 1, 1],
            &[
                norm.clone(),
                w.get("up.3.res.0.conv1.weight"),
                w.get("up.3.res.0.conv1.bias"),
                tmp.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "groupnorm_g32_c128_h1024_w1024_silu.comp",
            [32, 1, 1],
            &[
                tmp.clone(),
                w.get("up.3.res.0.norm2.gamma"),
                w.get("up.3.res.0.norm2.beta"),
                norm.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "conv2d_3x3_c128_f128_h1024_w1024_bias.comp",
            [128, 128, 1],
            &[
                norm.clone(),
                w.get("up.3.res.0.conv2.weight"),
                w.get("up.3.res.0.conv2.bias"),
                tmp.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "conv2d_1x1_c256_f128_h1024_w1024_bias_residual.comp",
            [1, 8192, 1],
            &[
                x.clone(),
                w.get("up.3.res.0.conv_shortcut.weight"),
                w.get("up.3.res.0.conv_shortcut.bias"),
                tmp.clone(),
            ],
        )
        .record(&mut cmd);
        swap(&mut x, &mut tmp);
        for r in 1..3 {
            resnet_same(
                &mut cmd,
                &format!("up.3.res.{r}"),
                "groupnorm_g32_c128_h1024_w1024_silu.comp",
                "conv2d_3x3_c128_f128_h1024_w1024_bias.comp",
                "conv2d_3x3_c128_f128_h1024_w1024_bias_residual.comp",
                [128, 128, 1],
                [8192, 1, 1],
                &x,
                &norm,
                &tmp,
            );
        }
        self.bind(
            "groupnorm_g32_c128_h1024_w1024_silu.comp",
            [32, 1, 1],
            &[
                x,
                w.get("norm_out.gamma"),
                w.get("norm_out.beta"),
                norm.clone(),
            ],
        )
        .record(&mut cmd);
        self.bind(
            "conv2d_3x3_c128_f3_h1024_w1024_bias_clamp_rgb8.comp",
            [128, 1024, 1],
            &[
                norm,
                w.get("conv_out.weight"),
                w.get("conv_out.bias"),
                rgb_out.clone(),
            ],
        )
        .record(&mut cmd);
        cmd.build().unwrap()
    }
}

fn encode_clip_tokens(tokenizer: &Tokenizer, prompt: &str, pad_id: u32) -> (Vec<u8>, u64) {
    let enc = tokenizer.encode(prompt, false).unwrap();
    let mut ids = Vec::with_capacity(77);
    ids.push(49406u32);
    for &id in enc.get_ids().iter().take(75) {
        if id != 49406 && id != 49407 {
            ids.push(id);
        }
    }
    let eos_idx = ids.len() as u64;
    ids.push(49407);
    ids.resize(77, pad_id);
    let bytes = ids.into_iter().flat_map(u32::to_le_bytes).collect();
    (bytes, eos_idx)
}

fn encode_t5_tokens(tokenizer: &Tokenizer, prompt: &str) -> Vec<u8> {
    let enc = tokenizer.encode(prompt, false).unwrap();
    let mut ids = Vec::with_capacity(256);
    for &id in enc.get_ids().iter().take(255) {
        if id != 1 {
            ids.push(id);
        }
    }
    ids.push(1u32);
    ids.resize(256, 0);
    ids.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn write_png_rgb8(path: &Path, w: u32, h: u32, rgb: &[u8]) {
    let file = BufWriter::new(File::create(path).unwrap());
    let mut encoder = png::Encoder::new(file, w, h);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(rgb).unwrap();
}

static STOP_FLAG: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sigterm(_: libc::c_int) {
    STOP_FLAG.store(true, Ordering::SeqCst);
}

pub(crate) fn get_data_dir() -> PathBuf {
    PathBuf::from(var("HOME").unwrap()).join(".local/share/halodiffuse")
}

fn get_sock_path() -> PathBuf {
    get_data_dir().join("halodiffuse.sock")
}

fn get_pid_path() -> PathBuf {
    get_data_dir().join("halodiffuse.pid")
}

fn read_urandom_u64() -> u64 {
    let mut f = File::open("/dev/urandom").unwrap();
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).unwrap();
    u64::from_le_bytes(buf)
}

fn print_usage() {
    let bin = env!("CARGO_PKG_NAME");
    println!(
        "{bin} {}\n{}\n\n\
         Usage:\n  \
           {bin} install [--force] [--raw-dir <DIR> | <DIR>]\n  \
           {bin} start\n  \
           {bin} stop\n  \
           {bin} status\n  \
           {bin} --prompt <TEXT> --path <FILE> [--seed <U64>]\n\n\
         Options:\n  \
           -p, --prompt <TEXT>  Text prompt\n  \
           -o, --path <FILE>    Output PNG path\n  \
           -s, --seed <U64>     RNG seed\n  \
           -h, --help           Show help\n  \
           -V, --version        Show version",
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_DESCRIPTION")
    );
}

fn run_daemon() {
    let data_dir = get_data_dir();
    let sock_path = get_sock_path();
    let pid_path = get_pid_path();
    let _ = remove_file(&sock_path);
    unsafe {
        libc::signal(
            libc::SIGTERM,
            handle_sigterm as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGINT,
            handle_sigterm as *const () as libc::sighandler_t,
        );
    }
    let engine = Engine::new();
    let tok_l = Tokenizer::from_bytes(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/tokenizer_l.json"
    )))
    .unwrap();
    let tok_g = Tokenizer::from_bytes(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/tokenizer_g.json"
    )))
    .unwrap();
    let tok_t5 = Tokenizer::from_bytes(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/tokenizer_t5.json"
    )))
    .unwrap();
    let w_clip_l;
    let w_clip_g;
    let w_t5;
    let w_mmdit;
    let w_vae;
    {
        let staging = engine.alloc_staging_bytes(256 * 1024 * 1024);
        w_clip_l = WeightPack::load(&engine, &staging, data_dir.join("clip_l.safetensors"));
        w_clip_g = WeightPack::load(&engine, &staging, data_dir.join("clip_g.safetensors"));
        w_t5 = WeightPack::load(&engine, &staging, data_dir.join("t5xxl.safetensors"));
        w_mmdit = WeightPack::load(&engine, &staging, data_dir.join("mmdit.safetensors"));
        w_vae = WeightPack::load(&engine, &staging, data_dir.join("vae.safetensors"));
    }
    let buf_tokens_l = engine.alloc_host_bytes(77 * 4);
    let buf_tokens_g = engine.alloc_host_bytes(77 * 4);
    let buf_tokens_t5 = engine.alloc_host_bytes(256 * 4);
    let latents_x = engine.alloc_device_bytes(4096 * 64 * 2);
    let num_latent_pairs = (latents_x.size() / 4) as u32;
    let clip_l_seq = engine.alloc_device_bytes(77 * 768 * 2);
    let clip_g_seq = engine.alloc_device_bytes(77 * 1280 * 2);
    let y_pooled = engine.alloc_device_bytes(2048 * 2);
    let ctx_333 = engine.alloc_device_bytes(333 * 4096 * 2);
    let v_pred = engine.alloc_device_bytes(4096 * 64 * 2);
    let latents_nchw = engine.alloc_device_bytes(16 * 128 * 128 * 2);
    let rgb_out = engine.alloc_host_bytes(1024 * 1024 * 3);
    let timesteps: [f32; 4] = [1000.0, 900.0, 750.0, 500.0];
    let dts: [f32; 4] = [-0.10, -0.15, -0.25, -0.50];
    let t_buffers: Vec<Subbuffer<[u8]>> = timesteps
        .iter()
        .map(|&t| {
            let b = engine.alloc_host_bytes(4);
            b.write().unwrap().copy_from_slice(&t.to_le_bytes());
            b
        })
        .collect();
    let scratch_arena = DeviceScratchArena::new(&engine);
    let text_ws = TextEncoderWorkspace::new(&scratch_arena);
    let mmdit_ws = MmditWorkspace::new(&scratch_arena);
    let vae_ws = VaeWorkspace::new(&engine, &scratch_arena);
    let prebuilt_text = engine.build_text_encoders(
        &text_ws,
        &w_clip_l,
        &w_clip_g,
        &w_t5,
        &latents_x,
        &buf_tokens_l,
        &buf_tokens_g,
        &buf_tokens_t5,
        &clip_l_seq,
        &clip_g_seq,
        &y_pooled,
        &ctx_333,
    );
    let mmdit_cmds: [Arc<PrimaryAutoCommandBuffer>; 4] = std::array::from_fn(|step| {
        let euler_nchw = if step == 3 { Some(&latents_nchw) } else { None };
        engine.record_mmdit_step_cmd(
            &mmdit_ws,
            &w_mmdit,
            &latents_x,
            &ctx_333,
            &y_pooled,
            &t_buffers[step],
            &v_pred,
            euler_nchw,
            dts[step],
        )
    });
    let vae_cmd = engine.record_vae_cmd(&vae_ws, &w_vae, &latents_nchw, &rgb_out);
    let listener = UnixListener::bind(&sock_path).unwrap();
    listener.set_nonblocking(true).unwrap();
    write(&pid_path, id().to_string()).unwrap();
    while !STOP_FLAG.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((mut stream, _)) => {
                if stream.set_nonblocking(false).is_err() {
                    continue;
                }
                let mut cmd = [0u8; 1];
                if stream.read_exact(&mut cmd).is_err() {
                    continue;
                }
                if cmd[0] == b'S' {
                    let _ = stream.write_all(b"Stopped halodiffuse daemon.\n");
                    break;
                } else if cmd[0] == b'G' {
                    let mut u64_buf = [0u8; 8];
                    let mut u32_buf = [0u8; 4];
                    if stream.read_exact(&mut u64_buf).is_err() {
                        continue;
                    }
                    let init_seed = u64::from_le_bytes(u64_buf);
                    if stream.read_exact(&mut u32_buf).is_err() {
                        continue;
                    }
                    let prompt_len = u32::from_le_bytes(u32_buf) as usize;
                    let mut prompt_bytes = vec![0u8; prompt_len];
                    if stream.read_exact(&mut prompt_bytes).is_err() {
                        continue;
                    }
                    let Ok(prompt) = String::from_utf8(prompt_bytes) else {
                        continue;
                    };
                    if stream.read_exact(&mut u32_buf).is_err() {
                        continue;
                    }
                    let path_len = u32::from_le_bytes(u32_buf) as usize;
                    let mut path_bytes = vec![0u8; path_len];
                    if stream.read_exact(&mut path_bytes).is_err() {
                        continue;
                    }
                    let Ok(path_str) = String::from_utf8(path_bytes) else {
                        continue;
                    };
                    let out_path = PathBuf::from(path_str);
                    let t_e2e = Instant::now();
                    let (clip_l_bytes, eos_l) = encode_clip_tokens(&tok_l, &prompt, 49407);
                    let (clip_g_bytes, eos_g) = encode_clip_tokens(&tok_g, &prompt, 0);
                    let t5_bytes = encode_t5_tokens(&tok_t5, &prompt);
                    buf_tokens_l.write().unwrap().copy_from_slice(&clip_l_bytes);
                    buf_tokens_g.write().unwrap().copy_from_slice(&clip_g_bytes);
                    buf_tokens_t5.write().unwrap().copy_from_slice(&t5_bytes);
                    let t_gpu = Instant::now();
                    engine.run_stage1_text_and_noise(
                        &prebuilt_text,
                        init_seed,
                        num_latent_pairs,
                        eos_l,
                        eos_g,
                    );
                    for step_cmd in &mmdit_cmds {
                        engine.submit_recorded_and_wait(step_cmd);
                    }
                    engine.submit_recorded_and_wait(&vae_cmd);
                    let gpu_secs = t_gpu.elapsed().as_secs_f32();
                    if let Some(parent) = out_path.parent() {
                        let _ = create_dir_all(parent);
                    }
                    let rgb_bytes = rgb_out.read().unwrap();
                    write_png_rgb8(&out_path, 1024, 1024, &rgb_bytes);
                    let msg = format!(
                        "Generated {} (1024x1024, seed {init_seed}) in {gpu_secs:.2}s (e2e {:.2}s)\n",
                        out_path.display(),
                        t_e2e.elapsed().as_secs_f32()
                    );
                    let _ = stream.write_all(msg.as_bytes());
                }
            }
            Err(ref e) if e.kind() == ErrorKind::WouldBlock => {
                sleep(Duration::from_millis(20));
            }
            Err(_) => break,
        }
    }
    let _ = remove_file(&sock_path);
    let _ = remove_file(&pid_path);
}

fn main() {
    let args: Vec<String> = args().skip(1).collect();
    let sock_path = get_sock_path();
    let pid_path = get_pid_path();
    if args.is_empty() {
        print_usage();
        exit(1);
    }
    match args[0].as_str() {
        "-h" | "--help" | "help" => print_usage(),
        "-V" | "--version" => {
            println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        }
        "install" => install::run(&args[1..]),
        "__daemon" => run_daemon(),
        "status" => {
            if UnixStream::connect(&sock_path).is_ok() {
                let pid = read_to_string(&pid_path).unwrap_or_default();
                println!("Running (PID {})", pid.trim());
            } else {
                println!("Stopped");
                exit(1);
            }
        }
        "start" => {
            let data_dir = get_data_dir();
            for pack in [
                "clip_l.safetensors",
                "clip_g.safetensors",
                "t5xxl.safetensors",
                "mmdit.safetensors",
                "vae.safetensors",
            ] {
                if !data_dir.join(pack).exists() {
                    eprintln!(
                        "Missing {}. Run `{} install` first.",
                        data_dir.join(pack).display(),
                        env!("CARGO_PKG_NAME")
                    );
                    exit(1);
                }
            }
            if UnixStream::connect(&sock_path).is_ok() {
                println!("halodiffuse is already running.");
                return;
            }
            let _ = remove_file(&sock_path);
            let _ = remove_file(&pid_path);
            let t0 = Instant::now();
            let mut child = Command::new(current_exe().unwrap())
                .arg("__daemon")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap();
            print!("Starting daemon (PID {})... ", child.id());
            stdout().flush().unwrap();
            while !sock_path.exists() {
                if let Some(status) = child.try_wait().unwrap() {
                    println!("failed");
                    eprintln!("Daemon exited early with status {status}");
                    exit(1);
                }
                sleep(Duration::from_millis(50));
            }
            println!("ready ({:.2}s)", t0.elapsed().as_secs_f32());
        }
        "stop" => {
            let mut stopped = false;
            if let Ok(mut stream) = UnixStream::connect(&sock_path) {
                let _ = stream.write_all(b"S");
                let mut resp = String::new();
                let _ = stream.read_to_string(&mut resp);
                if !resp.is_empty() {
                    print!("{resp}");
                }
                stopped = true;
            } else if let Ok(pid_str) = read_to_string(&pid_path)
                && let Ok(pid) = pid_str.trim().parse::<i32>()
            {
                unsafe {
                    if libc::kill(pid, libc::SIGTERM) == 0 {
                        println!("Stopped halodiffuse daemon (PID {pid}).");
                        stopped = true;
                    }
                }
            }
            let _ = remove_file(&sock_path);
            let _ = remove_file(&pid_path);
            if !stopped {
                eprintln!("halodiffuse is not running.");
                exit(1);
            }
        }
        _ => {
            let mut prompt: Option<String> = None;
            let mut out_path: Option<PathBuf> = None;
            let mut seed: Option<u64> = None;
            let mut i = 0;
            while i < args.len() {
                match args[i].as_str() {
                    "-p" | "--prompt" | "—prompt" => {
                        i += 1;
                        prompt = args.get(i).cloned();
                    }
                    "-o" | "--path" | "—path" => {
                        i += 1;
                        out_path = args.get(i).map(PathBuf::from);
                    }
                    "-s" | "--seed" | "—seed" => {
                        i += 1;
                        let Some(val) = args.get(i) else {
                            eprintln!("Error: missing value for --seed");
                            exit(1);
                        };
                        seed = Some(val.parse().unwrap_or_else(|_| {
                            eprintln!("Error: invalid u64 `{val}` for --seed");
                            exit(1);
                        }));
                    }
                    other => {
                        eprintln!("Error: unknown argument `{other}`");
                        exit(1);
                    }
                }
                i += 1;
            }
            let Some(prompt) = prompt else {
                eprintln!("Error: missing required --prompt <TEXT>");
                exit(1);
            };
            let Some(out_path) = out_path else {
                eprintln!("Error: missing required --path <FILE>");
                exit(1);
            };
            if out_path.is_dir() {
                eprintln!("Error: --path must be a file path, not a directory.");
                exit(1);
            }
            let abs_out_path = if out_path.is_absolute() {
                out_path
            } else {
                current_dir().unwrap().join(out_path)
            };
            let seed = seed.unwrap_or_else(read_urandom_u64);
            let mut stream = UnixStream::connect(&sock_path).unwrap_or_else(|_| {
                eprintln!(
                    "halodiffuse daemon is not running. Start it with: {} start",
                    env!("CARGO_PKG_NAME")
                );
                exit(1);
            });
            let prompt_bytes = prompt.as_bytes();
            let path_bytes = abs_out_path.to_str().unwrap().as_bytes();
            stream.write_all(b"G").unwrap();
            stream.write_all(&seed.to_le_bytes()).unwrap();
            stream
                .write_all(&(prompt_bytes.len() as u32).to_le_bytes())
                .unwrap();
            stream.write_all(prompt_bytes).unwrap();
            stream
                .write_all(&(path_bytes.len() as u32).to_le_bytes())
                .unwrap();
            stream.write_all(path_bytes).unwrap();
            stream.flush().unwrap();
            let mut reader = BufReader::new(stream);
            let mut resp = String::new();
            reader.read_line(&mut resp).unwrap();
            print!("{resp}");
        }
    }
}
