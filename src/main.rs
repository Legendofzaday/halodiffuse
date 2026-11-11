#![feature(f16)]

use std::{
    collections::HashMap,
    env,
    fs::{self, File},
    io::{BufRead, BufReader, Read, Write},
    os::unix::{
        fs::FileExt,
        net::{UnixListener, UnixStream},
    },
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;
use vulkano::{
    Version, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::GpuFuture,
};

#[repr(C)]
#[derive(Clone, Copy, BufferContents)]
struct EulerPushConstants {
    dt: f32,
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
    fn load(allocator: &Arc<StandardMemoryAllocator>, path: impl AsRef<Path>) -> Self {
        let mut file = File::open(path.as_ref()).unwrap();
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf).unwrap();
        let header_len = u64::from_le_bytes(len_buf) as usize;
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes).unwrap();
        let header = String::from_utf8(header_bytes).unwrap();
        let data_base = 8 + header_len as u64;
        let mut tensors = HashMap::new();
        let hbytes = header.as_bytes();
        let mut pos = 0;
        while pos < hbytes.len() {
            let Some(q1) = header[pos..].find('"') else {
                break;
            };
            let k_start = pos + q1 + 1;
            let q2 = header[k_start..].find('"').unwrap() + k_start;
            let key = &header[k_start..q2];
            let b_open = header[q2..].find('{').unwrap() + q2;
            let mut depth = 0;
            let mut b_close = b_open;
            for (i, &ch) in hbytes[b_open..].iter().enumerate() {
                if ch == b'{' {
                    depth += 1;
                } else if ch == b'}' {
                    depth -= 1;
                    if depth == 0 {
                        b_close = b_open + i;
                        break;
                    }
                }
            }
            if key != "__metadata__" {
                let body = &header[b_open..=b_close];
                let do_pos = body.find("\"data_offsets\"").unwrap();
                let br_open = body[do_pos..].find('[').unwrap() + do_pos + 1;
                let br_close = body[br_open..].find(']').unwrap() + br_open;
                let mut nums = body[br_open..br_close]
                    .split(',')
                    .map(|s| s.trim().parse::<u64>().unwrap());
                let start = nums.next().unwrap();
                let end = nums.next().unwrap();
                let buf = Buffer::new_slice::<u8>(
                    allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    end - start,
                )
                .unwrap();
                file.read_exact_at(&mut *buf.write().unwrap(), data_base + start)
                    .unwrap();
                tensors.insert(key.to_string(), buf);
            }
            pos = b_close + 1;
        }
        Self { tensors }
    }

    fn get(&self, key: &str) -> Subbuffer<[u8]> {
        self.tensors.get(key).unwrap().clone()
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
    fn new(engine: &Engine) -> Self {
        let d_bytes = 2432 * 2;
        Self {
            t_silu: engine.alloc_device_bytes(d_bytes),
            t_out: engine.alloc_device_bytes(d_bytes),
            y_silu: engine.alloc_device_bytes(d_bytes),
            c_silu: engine.alloc_device_bytes(d_bytes),
            x: engine.alloc_device_bytes(4096 * d_bytes),
            ctx: engine.alloc_device_bytes(333 * d_bytes),
            adaln_dual: engine.alloc_device_bytes(12 * d_bytes),
            adaln_2: engine.alloc_device_bytes(2 * d_bytes),
            x_norm: engine.alloc_device_bytes(4096 * d_bytes),
            ctx_norm: engine.alloc_device_bytes(333 * d_bytes),
            x_qkv: engine.alloc_device_bytes(4096 * 3 * d_bytes),
            ctx_qkv: engine.alloc_device_bytes(333 * 3 * d_bytes),
            qk_concat: engine.alloc_device_bytes(4429 * 2 * d_bytes),
            attn_concat: engine.alloc_device_bytes(4429 * d_bytes),
            x_mlp: engine.alloc_device_bytes(4096 * 4 * d_bytes),
            ctx_mlp: engine.alloc_device_bytes(333 * 4 * d_bytes),
        }
    }
}

struct Engine {
    device: Arc<Device>,
    queue: Arc<Queue>,
    mem_alloc: Arc<StandardMemoryAllocator>,
    desc_alloc: Arc<StandardDescriptorSetAllocator>,
    cmd_alloc: Arc<StandardCommandBufferAllocator>,
    pipelines: HashMap<&'static str, Arc<ComputePipeline>>,
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

    fn alloc_device_bytes(&self, size: u64) -> Subbuffer<[u8]> {
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

    fn alloc_host_bytes(&self, size: u64) -> Subbuffer<[u8]> {
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

    fn new_cmd(&self) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.cmd_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
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

    fn dispatch(
        &self,
        cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        shader: &'static str,
        grid: [u32; 3],
        buffers: &[Subbuffer<[u8]>],
        push_dt: Option<f32>,
    ) {
        let pipeline = self.pipelines.get(shader).unwrap();
        let layout = pipeline.layout();
        let set_layout = layout.set_layouts()[0].clone();
        let writes: Vec<WriteDescriptorSet> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| WriteDescriptorSet::buffer(i as u32, buf.clone()))
            .collect();
        let set = DescriptorSet::new(self.desc_alloc.clone(), set_layout, writes, []).unwrap();
        cmd.bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, layout.clone(), 0, set)
            .unwrap();
        if let Some(dt) = push_dt {
            cmd.push_constants(layout.clone(), 0, EulerPushConstants { dt })
                .unwrap();
        }
        unsafe {
            cmd.dispatch(grid).unwrap();
        }
    }

    fn record_clip(
        &self,
        cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        w: &WeightPack,
        tokens_buf: &Subbuffer<[u8]>,
        eos_index: u64,
        d: usize,
        num_layers: usize,
        out_seq: &Subbuffer<[u8]>,
        out_pooled: &Subbuffer<[u8]>,
    ) {
        let is_g = d == 1280;
        let bytes_seq = (77 * d * 2) as u64;
        let x = self.alloc_device_bytes(bytes_seq);
        let norm = self.alloc_device_bytes(bytes_seq);
        let qkv = self.alloc_device_bytes(bytes_seq * 3);
        let attn_out = self.alloc_device_bytes(bytes_seq);
        let mlp_hid = self.alloc_device_bytes(bytes_seq * 4);
        let (sh_emb, sh_ln, sh_qkv, sh_fa, sh_out, sh_fc1, sh_fc2, sh_eos) = if is_g {
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
        self.dispatch(
            cmd,
            sh_emb,
            [77, 1, 1],
            &[
                tokens_buf.clone(),
                w.get("embeddings.token_embedding.weight"),
                w.get("embeddings.position_embedding.weight"),
                x.clone(),
            ],
            None,
        );
        for i in 0..num_layers {
            self.dispatch(
                cmd,
                sh_ln,
                [77, 1, 1],
                &[
                    x.clone(),
                    w.get(&format!("layers.{i}.ln1.gamma")),
                    w.get(&format!("layers.{i}.ln1.beta")),
                    norm.clone(),
                ],
                None,
            );
            self.dispatch(
                cmd,
                sh_qkv,
                [nx_d * 3, 3, 1],
                &[
                    norm.clone(),
                    w.get(&format!("layers.{i}.attn.qkv.weight")),
                    w.get(&format!("layers.{i}.attn.qkv.bias")),
                    qkv.clone(),
                ],
                None,
            );
            self.dispatch(
                cmd,
                sh_fa,
                [1, heads, 1],
                &[qkv.clone(), attn_out.clone()],
                None,
            );
            self.dispatch(
                cmd,
                sh_out,
                [nx_d, 3, 1],
                &[
                    attn_out.clone(),
                    w.get(&format!("layers.{i}.attn.out_proj.weight")),
                    w.get(&format!("layers.{i}.attn.out_proj.bias")),
                    x.clone(),
                ],
                None,
            );
            self.dispatch(
                cmd,
                sh_ln,
                [77, 1, 1],
                &[
                    x.clone(),
                    w.get(&format!("layers.{i}.ln2.gamma")),
                    w.get(&format!("layers.{i}.ln2.beta")),
                    norm.clone(),
                ],
                None,
            );
            self.dispatch(
                cmd,
                sh_fc1,
                [nx_d * 4, 3, 1],
                &[
                    norm.clone(),
                    w.get(&format!("layers.{i}.mlp.fc1.weight")),
                    w.get(&format!("layers.{i}.mlp.fc1.bias")),
                    mlp_hid.clone(),
                ],
                None,
            );
            self.dispatch(
                cmd,
                sh_fc2,
                [nx_d, 3, 1],
                &[
                    mlp_hid.clone(),
                    w.get(&format!("layers.{i}.mlp.fc2.weight")),
                    w.get(&format!("layers.{i}.mlp.fc2.bias")),
                    x.clone(),
                ],
                None,
            );
            if i == num_layers - 2 {
                cmd.copy_buffer(CopyBufferInfo::buffers(x.clone(), out_seq.clone()))
                    .unwrap();
            }
        }
        let row_bytes = (d * 2) as u64;
        let eos_slice = x.slice((eos_index * row_bytes)..((eos_index + 1) * row_bytes));
        self.dispatch(
            cmd,
            sh_eos,
            [nx_d, 1, 1],
            &[
                eos_slice,
                w.get("final_ln.gamma"),
                w.get("final_ln.beta"),
                w.get("eos_proj.weight"),
                w.get("eos_proj.bias"),
                out_pooled.clone(),
            ],
            None,
        );
    }

    fn record_t5(
        &self,
        cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        w: &WeightPack,
        tokens_buf: &Subbuffer<[u8]>,
        out_t5: &Subbuffer<[u8]>,
    ) {
        let bytes_seq = 256 * 4096 * 2;
        let x = self.alloc_device_bytes(bytes_seq);
        let norm = self.alloc_device_bytes(bytes_seq);
        let qkv = self.alloc_device_bytes(bytes_seq * 3);
        let attn_out = self.alloc_device_bytes(bytes_seq);
        let ffn_hid = self.alloc_device_bytes(256 * 10240 * 2);
        self.dispatch(
            cmd,
            "embed_lookup_v32128_m256_d4096.comp",
            [256, 1, 1],
            &[tokens_buf.clone(), w.get("embed_tokens.weight"), x.clone()],
            None,
        );
        let rpb = w.get("rpb_weight");
        for i in 0..24 {
            self.dispatch(
                cmd,
                "rmsnorm_m256_d4096.comp",
                [256, 1, 1],
                &[
                    x.clone(),
                    w.get(&format!("blocks.{i}.ln1.gamma")),
                    norm.clone(),
                ],
                None,
            );
            self.dispatch(
                cmd,
                "hgemm_m256_n12288_k4096.comp",
                [96, 2, 1],
                &[
                    norm.clone(),
                    w.get(&format!("blocks.{i}.attn.qkv.weight")),
                    qkv.clone(),
                ],
                None,
            );
            self.dispatch(
                cmd,
                "flash_attention_rpb_m256_h64_d64.comp",
                [4, 64, 1],
                &[qkv.clone(), rpb.clone(), attn_out.clone()],
                None,
            );
            self.dispatch(
                cmd,
                "hgemm_m256_n4096_k4096_residual.comp",
                [32, 2, 1],
                &[
                    attn_out.clone(),
                    w.get(&format!("blocks.{i}.attn.o.weight")),
                    x.clone(),
                ],
                None,
            );
            self.dispatch(
                cmd,
                "rmsnorm_m256_d4096.comp",
                [256, 1, 1],
                &[
                    x.clone(),
                    w.get(&format!("blocks.{i}.ln2.gamma")),
                    norm.clone(),
                ],
                None,
            );
            self.dispatch(
                cmd,
                "hgemm_m256_n10240_k4096_geglu.comp",
                [160, 2, 1],
                &[
                    norm.clone(),
                    w.get(&format!("blocks.{i}.ffn.b0_wi0.weight")),
                    w.get(&format!("blocks.{i}.ffn.b1_wi1.weight")),
                    ffn_hid.clone(),
                ],
                None,
            );
            self.dispatch(
                cmd,
                "hgemm_m256_n4096_k10240_residual.comp",
                [32, 2, 1],
                &[
                    ffn_hid.clone(),
                    w.get(&format!("blocks.{i}.ffn.wo.weight")),
                    x.clone(),
                ],
                None,
            );
        }
        self.dispatch(
            cmd,
            "rmsnorm_m256_d4096.comp",
            [256, 1, 1],
            &[x, w.get("final_ln.gamma"), out_t5.clone()],
            None,
        );
    }

    fn run_mmdit_step(
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
    ) {
        let d_bytes = 2432 * 2;
        let chunk = |buf: &Subbuffer<[u8]>, idx: u64| -> Subbuffer<[u8]> {
            buf.clone().slice((idx * d_bytes)..((idx + 1) * d_bytes))
        };
        let mut cmd1 = self.new_cmd();
        self.dispatch(
            &mut cmd1,
            "timestep_embed_sinusoidal_hgemm_m1_n2432_k256_bias_silu.comp",
            [19, 1, 1],
            &[
                t_buf.clone(),
                w.get("t_embedder.mlp.0.weight"),
                w.get("t_embedder.mlp.0.bias"),
                ws.t_silu.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd1,
            "hgemm_m1_n2432_k2432_bias.comp",
            [19, 1, 1],
            &[
                ws.t_silu.clone(),
                w.get("t_embedder.mlp.2.weight"),
                w.get("t_embedder.mlp.2.bias"),
                ws.t_out.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd1,
            "hgemm_m1_n2432_k2048_bias_silu.comp",
            [19, 1, 1],
            &[
                y_pooled.clone(),
                w.get("y_embedder.mlp.0.weight"),
                w.get("y_embedder.mlp.0.bias"),
                ws.y_silu.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd1,
            "hgemm_m1_n2432_k2432_bias_cond_add_silu.comp",
            [19, 1, 1],
            &[
                ws.y_silu.clone(),
                w.get("y_embedder.mlp.2.weight"),
                w.get("y_embedder.mlp.2.bias"),
                ws.t_out.clone(),
                ws.c_silu.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd1,
            "patch_embed_conv2d_m4096_n2432_k64.comp",
            [19, 32, 1],
            &[
                latents_x.clone(),
                w.get("x_embedder.proj.weight"),
                w.get("x_embedder.fused_pos_bias"),
                ws.x.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd1,
            "hgemm_m333_n2432_k4096_bias.comp",
            [19, 11, 1],
            &[
                ctx_in.clone(),
                w.get("context_embedder.weight"),
                w.get("context_embedder.bias"),
                ws.ctx.clone(),
            ],
            None,
        );
        let record_blocks = |cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                             range: std::ops::Range<usize>| {
            for i in range {
                let is_last = i == 37;
                if !is_last {
                    self.dispatch(
                        cmd,
                        "adaln_mod_m1_n14592_k2432_bias.comp",
                        [114, 2, 1],
                        &[
                            ws.c_silu.clone(),
                            w.get(&format!("blocks.{i}.adaln_14592.weight")),
                            w.get(&format!("blocks.{i}.adaln_14592.bias")),
                            ws.adaln_dual.clone(),
                        ],
                        None,
                    );
                } else {
                    self.dispatch(
                        cmd,
                        "adaln_mod_m1_n14592_k2432_bias.comp",
                        [114, 1, 1],
                        &[
                            ws.c_silu.clone(),
                            w.get(&format!("blocks.{i}.x_adaln_14592.weight")),
                            w.get(&format!("blocks.{i}.x_adaln_14592.bias")),
                            ws.adaln_dual.clone(),
                        ],
                        None,
                    );
                    self.dispatch(
                        cmd,
                        "adaln_mod_m1_n4864_k2432_bias.comp",
                        [38, 1, 1],
                        &[
                            ws.c_silu.clone(),
                            w.get(&format!("blocks.{i}.ctx_adaln_4864.weight")),
                            w.get(&format!("blocks.{i}.ctx_adaln_4864.bias")),
                            ws.adaln_2.clone(),
                        ],
                        None,
                    );
                }
                self.dispatch(
                    cmd,
                    "layernorm_adaln_m4096_d2432.comp",
                    [4096, 1, 1],
                    &[
                        ws.x.clone(),
                        chunk(&ws.adaln_dual, 1),
                        chunk(&ws.adaln_dual, 0),
                        ws.x_norm.clone(),
                    ],
                    None,
                );
                let (ctx_gamma_msa, ctx_beta_msa) = if !is_last {
                    (chunk(&ws.adaln_dual, 7), chunk(&ws.adaln_dual, 6))
                } else {
                    (chunk(&ws.adaln_2, 1), chunk(&ws.adaln_2, 0))
                };
                self.dispatch(
                    cmd,
                    "layernorm_adaln_m333_d2432.comp",
                    [333, 1, 1],
                    &[
                        ws.ctx.clone(),
                        ctx_gamma_msa,
                        ctx_beta_msa,
                        ws.ctx_norm.clone(),
                    ],
                    None,
                );
                self.dispatch(
                    cmd,
                    "hgemm_m4096_n7296_k2432_bias.comp",
                    [57, 32, 1],
                    &[
                        ws.x_norm.clone(),
                        w.get(&format!("blocks.{i}.x_attn.qkv.weight")),
                        w.get(&format!("blocks.{i}.x_attn.qkv.bias")),
                        ws.x_qkv.clone(),
                    ],
                    None,
                );
                self.dispatch(
                    cmd,
                    "hgemm_m333_n7296_k2432_bias.comp",
                    [57, 11, 1],
                    &[
                        ws.ctx_norm.clone(),
                        w.get(&format!("blocks.{i}.ctx_attn.qkv.weight")),
                        w.get(&format!("blocks.{i}.ctx_attn.qkv.bias")),
                        ws.ctx_qkv.clone(),
                    ],
                    None,
                );
                self.dispatch(
                    cmd,
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
                    None,
                );
                self.dispatch(
                    cmd,
                    "flash_attention_v_concat_m4429_h38_d64.comp",
                    [70, 38, 1],
                    &[
                        ws.qk_concat.clone(),
                        ws.x_qkv.clone(),
                        ws.ctx_qkv.clone(),
                        ws.attn_concat.clone(),
                    ],
                    None,
                );
                let x_attn_slice = ws.attn_concat.clone().slice(0..(4096 * d_bytes));
                let ctx_attn_slice = ws
                    .attn_concat
                    .clone()
                    .slice((4096 * d_bytes)..(4429 * d_bytes));
                self.dispatch(
                    cmd,
                    "hgemm_m4096_n2432_k2432_bias_gated_residual.comp",
                    [19, 32, 1],
                    &[
                        x_attn_slice,
                        w.get(&format!("blocks.{i}.x_attn.proj.weight")),
                        w.get(&format!("blocks.{i}.x_attn.proj.bias")),
                        chunk(&ws.adaln_dual, 2),
                        ws.x.clone(),
                    ],
                    None,
                );
                self.dispatch(
                    cmd,
                    "layernorm_adaln_m4096_d2432.comp",
                    [4096, 1, 1],
                    &[
                        ws.x.clone(),
                        chunk(&ws.adaln_dual, 4),
                        chunk(&ws.adaln_dual, 3),
                        ws.x_norm.clone(),
                    ],
                    None,
                );
                self.dispatch(
                    cmd,
                    "hgemm_m4096_n9728_k2432_bias_gelu.comp",
                    [76, 32, 1],
                    &[
                        ws.x_norm.clone(),
                        w.get(&format!("blocks.{i}.x_mlp.fc1.weight")),
                        w.get(&format!("blocks.{i}.x_mlp.fc1.bias")),
                        ws.x_mlp.clone(),
                    ],
                    None,
                );
                self.dispatch(
                    cmd,
                    "hgemm_m4096_n2432_k9728_bias_gated_residual.comp",
                    [19, 32, 1],
                    &[
                        ws.x_mlp.clone(),
                        w.get(&format!("blocks.{i}.x_mlp.fc2.weight")),
                        w.get(&format!("blocks.{i}.x_mlp.fc2.bias")),
                        chunk(&ws.adaln_dual, 5),
                        ws.x.clone(),
                    ],
                    None,
                );
                if !is_last {
                    self.dispatch(
                        cmd,
                        "hgemm_m333_n2432_k2432_bias_gated_residual.comp",
                        [19, 11, 1],
                        &[
                            ctx_attn_slice,
                            w.get(&format!("blocks.{i}.ctx_attn.proj.weight")),
                            w.get(&format!("blocks.{i}.ctx_attn.proj.bias")),
                            chunk(&ws.adaln_dual, 8),
                            ws.ctx.clone(),
                        ],
                        None,
                    );
                    self.dispatch(
                        cmd,
                        "layernorm_adaln_m333_d2432.comp",
                        [333, 1, 1],
                        &[
                            ws.ctx.clone(),
                            chunk(&ws.adaln_dual, 10),
                            chunk(&ws.adaln_dual, 9),
                            ws.ctx_norm.clone(),
                        ],
                        None,
                    );
                    self.dispatch(
                        cmd,
                        "hgemm_m333_n9728_k2432_bias_gelu.comp",
                        [76, 11, 1],
                        &[
                            ws.ctx_norm.clone(),
                            w.get(&format!("blocks.{i}.ctx_mlp.fc1.weight")),
                            w.get(&format!("blocks.{i}.ctx_mlp.fc1.bias")),
                            ws.ctx_mlp.clone(),
                        ],
                        None,
                    );
                    self.dispatch(
                        cmd,
                        "hgemm_m333_n2432_k9728_bias_gated_residual.comp",
                        [19, 11, 1],
                        &[
                            ws.ctx_mlp.clone(),
                            w.get(&format!("blocks.{i}.ctx_mlp.fc2.weight")),
                            w.get(&format!("blocks.{i}.ctx_mlp.fc2.bias")),
                            chunk(&ws.adaln_dual, 11),
                            ws.ctx.clone(),
                        ],
                        None,
                    );
                }
            }
        };
        record_blocks(&mut cmd1, 0..19);
        self.submit_and_wait(cmd1);
        let mut cmd2 = self.new_cmd();
        record_blocks(&mut cmd2, 19..38);
        self.dispatch(
            &mut cmd2,
            "adaln_mod_m1_n4864_k2432_bias.comp",
            [38, 1, 1],
            &[
                ws.c_silu.clone(),
                w.get("final_layer.adaln_4864.weight"),
                w.get("final_layer.adaln_4864.bias"),
                ws.adaln_2.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd2,
            "layernorm_adaln_m4096_d2432.comp",
            [4096, 1, 1],
            &[
                ws.x.clone(),
                chunk(&ws.adaln_2, 1),
                chunk(&ws.adaln_2, 0),
                ws.x_norm.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd2,
            "hgemm_m4096_n64_k2432_bias.comp",
            [32, 1, 1],
            &[
                ws.x_norm.clone(),
                w.get("final_layer.linear.weight"),
                w.get("final_layer.linear.bias"),
                v_out.clone(),
            ],
            None,
        );
        if let Some(nchw_out) = euler_out_nchw {
            self.dispatch(
                &mut cmd2,
                "unpatchify_m4096_k64_to_latents_c16_h128_w128_euler_scale.comp",
                [256, 1, 1],
                &[v_out.clone(), latents_x.clone(), nchw_out.clone()],
                Some(dt),
            );
        } else {
            self.dispatch(
                &mut cmd2,
                "flow_match_euler_step_m4096_k64.comp",
                [1024, 1, 1],
                &[v_out.clone(), latents_x.clone()],
                Some(dt),
            );
        }
        self.submit_and_wait(cmd2);
    }

    fn run_vae(&self, w: &WeightPack, latents_nchw: &Subbuffer<[u8]>, rgb_out: &Subbuffer<[u8]>) {
        let max_bytes = 1024 * 1024 * 256 * 2;
        let mut x = self.alloc_device_bytes(max_bytes);
        let norm = self.alloc_device_bytes(max_bytes);
        let mut tmp = self.alloc_device_bytes(max_bytes);
        let qkv_128 = self.alloc_device_bytes(128 * 128 * 1536 * 2);
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
            self.dispatch(
                cmd,
                gn_sh,
                [32, 1, 1],
                &[
                    x_buf.clone(),
                    w.get(&format!("{p}.norm1.gamma")),
                    w.get(&format!("{p}.norm1.beta")),
                    n_buf.clone(),
                ],
                None,
            );
            self.dispatch(
                cmd,
                c1_sh,
                c1_grid,
                &[
                    n_buf.clone(),
                    w.get(&format!("{p}.conv1.weight")),
                    w.get(&format!("{p}.conv1.bias")),
                    t_buf.clone(),
                ],
                None,
            );
            self.dispatch(
                cmd,
                gn_sh,
                [32, 1, 1],
                &[
                    t_buf.clone(),
                    w.get(&format!("{p}.norm2.gamma")),
                    w.get(&format!("{p}.norm2.beta")),
                    n_buf.clone(),
                ],
                None,
            );
            self.dispatch(
                cmd,
                c2_res_sh,
                c2_grid,
                &[
                    n_buf.clone(),
                    w.get(&format!("{p}.conv2.weight")),
                    w.get(&format!("{p}.conv2.bias")),
                    x_buf.clone(),
                ],
                None,
            );
        };
        let mut cmd = self.new_cmd();
        self.dispatch(
            &mut cmd,
            "conv2d_3x3_c16_f512_h128_w128_bias.comp",
            [4, 128, 1],
            &[
                latents_nchw.clone(),
                w.get("conv_in.weight"),
                w.get("conv_in.bias"),
                x.clone(),
            ],
            None,
        );
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
        self.dispatch(
            &mut cmd,
            "groupnorm_g32_c512_h128_w128.comp",
            [32, 1, 1],
            &[
                x.clone(),
                w.get("mid.attn.norm.gamma"),
                w.get("mid.attn.norm.beta"),
                norm.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd,
            "conv2d_1x1_c512_f1536_h128_w128_bias.comp",
            [12, 128, 1],
            &[
                norm.clone(),
                w.get("mid.attn.qkv.weight"),
                w.get("mid.attn.qkv.bias"),
                qkv_128.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd,
            "flash_attention_m16384_h1_d512.comp",
            [1024, 1, 1],
            &[qkv_128, tmp.clone()],
            None,
        );
        self.dispatch(
            &mut cmd,
            "conv2d_1x1_c512_f512_h128_w128_bias_residual.comp",
            [4, 128, 1],
            &[
                tmp.clone(),
                w.get("mid.attn.proj_out.weight"),
                w.get("mid.attn.proj_out.bias"),
                x.clone(),
            ],
            None,
        );
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
        self.dispatch(
            &mut cmd,
            "conv2d_3x3_upsample2x_c512_f512_h256_w256_bias.comp",
            [4, 512, 1],
            &[
                x.clone(),
                w.get("up.0.upsample_conv.weight"),
                w.get("up.0.upsample_conv.bias"),
                tmp.clone(),
            ],
            None,
        );
        std::mem::swap(&mut x, &mut tmp);
        self.submit_and_wait(cmd);
        let mut cmd = self.new_cmd();
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
        self.dispatch(
            &mut cmd,
            "conv2d_3x3_upsample2x_c512_f512_h512_w512_bias.comp",
            [4, 2048, 1],
            &[
                x.clone(),
                w.get("up.1.upsample_conv.weight"),
                w.get("up.1.upsample_conv.bias"),
                tmp.clone(),
            ],
            None,
        );
        std::mem::swap(&mut x, &mut tmp);
        self.submit_and_wait(cmd);
        let mut cmd = self.new_cmd();
        self.dispatch(
            &mut cmd,
            "groupnorm_g32_c512_h512_w512_silu.comp",
            [32, 1, 1],
            &[
                x.clone(),
                w.get("up.2.res.0.norm1.gamma"),
                w.get("up.2.res.0.norm1.beta"),
                norm.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd,
            "conv2d_3x3_c512_f256_h512_w512_bias.comp",
            [2, 2048, 1],
            &[
                norm.clone(),
                w.get("up.2.res.0.conv1.weight"),
                w.get("up.2.res.0.conv1.bias"),
                tmp.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd,
            "groupnorm_g32_c256_h512_w512_silu.comp",
            [32, 1, 1],
            &[
                tmp.clone(),
                w.get("up.2.res.0.norm2.gamma"),
                w.get("up.2.res.0.norm2.beta"),
                norm.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd,
            "conv2d_3x3_c256_f256_h512_w512_bias.comp",
            [2, 2048, 1],
            &[
                norm.clone(),
                w.get("up.2.res.0.conv2.weight"),
                w.get("up.2.res.0.conv2.bias"),
                tmp.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd,
            "conv2d_1x1_c512_f256_h512_w512_bias_residual.comp",
            [2, 2048, 1],
            &[
                x.clone(),
                w.get("up.2.res.0.conv_shortcut.weight"),
                w.get("up.2.res.0.conv_shortcut.bias"),
                tmp.clone(),
            ],
            None,
        );
        std::mem::swap(&mut x, &mut tmp);
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
        self.dispatch(
            &mut cmd,
            "conv2d_3x3_upsample2x_c256_f256_h1024_w1024_bias.comp",
            [2, 8192, 1],
            &[
                x.clone(),
                w.get("up.2.upsample_conv.weight"),
                w.get("up.2.upsample_conv.bias"),
                tmp.clone(),
            ],
            None,
        );
        std::mem::swap(&mut x, &mut tmp);
        self.submit_and_wait(cmd);
        let mut cmd = self.new_cmd();
        self.dispatch(
            &mut cmd,
            "groupnorm_g32_c256_h1024_w1024_silu.comp",
            [32, 1, 1],
            &[
                x.clone(),
                w.get("up.3.res.0.norm1.gamma"),
                w.get("up.3.res.0.norm1.beta"),
                norm.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd,
            "conv2d_3x3_c256_f128_h1024_w1024_bias.comp",
            [16384, 1, 1],
            &[
                norm.clone(),
                w.get("up.3.res.0.conv1.weight"),
                w.get("up.3.res.0.conv1.bias"),
                tmp.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd,
            "groupnorm_g32_c128_h1024_w1024_silu.comp",
            [32, 1, 1],
            &[
                tmp.clone(),
                w.get("up.3.res.0.norm2.gamma"),
                w.get("up.3.res.0.norm2.beta"),
                norm.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd,
            "conv2d_3x3_c128_f128_h1024_w1024_bias.comp",
            [128, 128, 1],
            &[
                norm.clone(),
                w.get("up.3.res.0.conv2.weight"),
                w.get("up.3.res.0.conv2.bias"),
                tmp.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd,
            "conv2d_1x1_c256_f128_h1024_w1024_bias_residual.comp",
            [1, 8192, 1],
            &[
                x.clone(),
                w.get("up.3.res.0.conv_shortcut.weight"),
                w.get("up.3.res.0.conv_shortcut.bias"),
                tmp.clone(),
            ],
            None,
        );
        std::mem::swap(&mut x, &mut tmp);
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
        self.dispatch(
            &mut cmd,
            "groupnorm_g32_c128_h1024_w1024_silu.comp",
            [32, 1, 1],
            &[
                x,
                w.get("norm_out.gamma"),
                w.get("norm_out.beta"),
                norm.clone(),
            ],
            None,
        );
        self.dispatch(
            &mut cmd,
            "conv2d_3x3_c128_f3_h1024_w1024_bias_clamp_rgb8.comp",
            [128, 1024, 1],
            &[
                norm,
                w.get("conv_out.weight"),
                w.get("conv_out.bias"),
                rgb_out.clone(),
            ],
            None,
        );
        self.submit_and_wait(cmd);
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

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn write_png_rgb8(path: &Path, w: u32, h: u32, rgb: &[u8]) {
    let mut out = Vec::with_capacity(rgb.len() + 65536);
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut write_chunk = |tag: &[u8; 4], payload: &[u8]| {
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        let start = out.len();
        out.extend_from_slice(tag);
        out.extend_from_slice(payload);
        let crc = crc32(&out[start..]);
        out.extend_from_slice(&crc.to_be_bytes());
    };
    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&w.to_be_bytes());
    ihdr[4..8].copy_from_slice(&h.to_be_bytes());
    ihdr[8] = 8;
    ihdr[9] = 2;
    write_chunk(b"IHDR", &ihdr);
    let row_bytes = (w as usize) * 3;
    let mut raw = Vec::with_capacity((row_bytes + 1) * (h as usize));
    for row in rgb.chunks_exact(row_bytes) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut idat = Vec::with_capacity(raw.len() + (raw.len() / 65535 + 2) * 5 + 6);
    idat.extend_from_slice(&[0x78, 0x01]);
    let mut s1 = 1u32;
    let mut s2 = 0u32;
    let chunks: Vec<&[u8]> = raw.chunks(65535).collect();
    for (i, block) in chunks.iter().enumerate() {
        let is_last = if i + 1 == chunks.len() { 1u8 } else { 0u8 };
        let len = block.len() as u16;
        idat.push(is_last);
        idat.extend_from_slice(&len.to_le_bytes());
        idat.extend_from_slice(&(!len).to_le_bytes());
        idat.extend_from_slice(block);
        for &b in *block {
            s1 = (s1 + b as u32) % 65521;
            s2 = (s2 + s1) % 65521;
        }
    }
    idat.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());
    write_chunk(b"IDAT", &idat);
    write_chunk(b"IEND", &[]);
    File::create(path).unwrap().write_all(&out).unwrap();
}

static STOP_FLAG: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sigterm(_: i32) {
    STOP_FLAG.store(true, Ordering::SeqCst);
}

fn get_data_dir() -> PathBuf {
    PathBuf::from(env::var("HOME").unwrap()).join(".local/share/halodiffuse")
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

fn run_daemon() {
    let data_dir = get_data_dir();
    let sock_path = get_sock_path();
    let pid_path = get_pid_path();
    let _ = fs::remove_file(&sock_path);

    unsafe {
        unsafe extern "C" {
            fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
        }
        signal(15, handle_sigterm); // SIGTERM
        signal(2, handle_sigterm); // SIGINT
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
    let w_clip_l = WeightPack::load(&engine.mem_alloc, data_dir.join("clip_l.safetensors"));
    let w_clip_g = WeightPack::load(&engine.mem_alloc, data_dir.join("clip_g.safetensors"));
    let w_t5 = WeightPack::load(&engine.mem_alloc, data_dir.join("t5xxl.safetensors"));
    let w_mmdit = WeightPack::load(&engine.mem_alloc, data_dir.join("mmdit.safetensors"));
    let w_vae = WeightPack::load(&engine.mem_alloc, data_dir.join("vae.safetensors"));
    let buf_tokens_l = engine.alloc_host_bytes(77 * 4);
    let buf_tokens_g = engine.alloc_host_bytes(77 * 4);
    let buf_tokens_t5 = engine.alloc_host_bytes(256 * 4);
    let latents_x = engine.alloc_host_bytes(4096 * 64 * 2);
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
    let mmdit_ws = MmditWorkspace::new(&engine);
    let listener = UnixListener::bind(&sock_path).unwrap();
    listener.set_nonblocking(true).unwrap();
    fs::write(&pid_path, std::process::id().to_string()).unwrap();
    while !STOP_FLAG.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((mut stream, _)) => {
                stream.set_nonblocking(false).unwrap();
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
                    stream.read_exact(&mut u64_buf).unwrap();
                    let mut seed = u64::from_le_bytes(u64_buf);
                    stream.read_exact(&mut u32_buf).unwrap();
                    let prompt_len = u32::from_le_bytes(u32_buf) as usize;
                    let mut prompt_bytes = vec![0u8; prompt_len];
                    stream.read_exact(&mut prompt_bytes).unwrap();
                    let prompt = String::from_utf8(prompt_bytes).unwrap();
                    stream.read_exact(&mut u32_buf).unwrap();
                    let path_len = u32::from_le_bytes(u32_buf) as usize;
                    let mut path_bytes = vec![0u8; path_len];
                    stream.read_exact(&mut path_bytes).unwrap();
                    let out_path = PathBuf::from(String::from_utf8(path_bytes).unwrap());
                    let (clip_l_bytes, eos_l) = encode_clip_tokens(&tok_l, &prompt, 49407);
                    let (clip_g_bytes, eos_g) = encode_clip_tokens(&tok_g, &prompt, 0);
                    let t5_bytes = encode_t5_tokens(&tok_t5, &prompt);
                    buf_tokens_l.write().unwrap().copy_from_slice(&clip_l_bytes);
                    buf_tokens_g.write().unwrap().copy_from_slice(&clip_g_bytes);
                    buf_tokens_t5.write().unwrap().copy_from_slice(&t5_bytes);
                    {
                        let mut guard = latents_x.write().unwrap();
                        let mut next_u64 = || {
                            seed = seed.wrapping_add(0x9E3779B97F4A7C15);
                            let mut z = seed;
                            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                            z ^ (z >> 31)
                        };
                        for chunk in guard.chunks_exact_mut(4) {
                            let u1 = ((next_u64() >> 11) as f64 + 0.5) / ((1u64 << 53) as f64);
                            let u2 = ((next_u64() >> 11) as f64) / ((1u64 << 53) as f64);
                            let r = (-2.0 * u1.ln()).sqrt();
                            let theta = 2.0 * std::f64::consts::PI * u2;
                            let b0 = ((r * theta.cos()) as f16).to_le_bytes();
                            let b1 = ((r * theta.sin()) as f16).to_le_bytes();
                            chunk[0] = b0[0];
                            chunk[1] = b0[1];
                            chunk[2] = b1[0];
                            chunk[3] = b1[1];
                        }
                    }
                    let t_start = Instant::now();
                    {
                        let mut cmd = engine.new_cmd();
                        let y_l_slice = y_pooled.clone().slice(0..(768 * 2));
                        let y_g_slice = y_pooled.clone().slice((768 * 2)..(2048 * 2));
                        engine.record_clip(
                            &mut cmd,
                            &w_clip_l,
                            &buf_tokens_l,
                            eos_l,
                            768,
                            12,
                            &clip_l_seq,
                            &y_l_slice,
                        );
                        engine.record_clip(
                            &mut cmd,
                            &w_clip_g,
                            &buf_tokens_g,
                            eos_g,
                            1280,
                            32,
                            &clip_g_seq,
                            &y_g_slice,
                        );
                        let ctx_clip_slice = ctx_333.clone().slice(0..(77 * 4096 * 2));
                        engine.dispatch(
                            &mut cmd,
                            "clip_concat_pad_m77_d4096.comp",
                            [77, 1, 1],
                            &[clip_l_seq.clone(), clip_g_seq.clone(), ctx_clip_slice],
                            None,
                        );
                        engine.submit_and_wait(cmd);
                        let mut cmd_t5 = engine.new_cmd();
                        let ctx_t5_slice = ctx_333.clone().slice((77 * 4096 * 2)..(333 * 4096 * 2));
                        engine.record_t5(&mut cmd_t5, &w_t5, &buf_tokens_t5, &ctx_t5_slice);
                        engine.submit_and_wait(cmd_t5);
                    }
                    for step in 0..4 {
                        let euler_nchw = if step == 3 { Some(&latents_nchw) } else { None };
                        engine.run_mmdit_step(
                            &mmdit_ws,
                            &w_mmdit,
                            &latents_x,
                            &ctx_333,
                            &y_pooled,
                            &t_buffers[step],
                            &v_pred,
                            euler_nchw,
                            dts[step],
                        );
                    }
                    engine.run_vae(&w_vae, &latents_nchw, &rgb_out);
                    let rgb_bytes = rgb_out.read().unwrap();
                    write_png_rgb8(&out_path, 1024, 1024, &rgb_bytes);
                    let msg = format!(
                        "Generated {} in {:.2?}\n",
                        out_path.display(),
                        t_start.elapsed()
                    );
                    let _ = stream.write_all(msg.as_bytes());
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(_) => break,
        }
    }
    let _ = fs::remove_file(&sock_path);
    let _ = fs::remove_file(&pid_path);
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!(
            "Usage:\n  halodiffuse start\n  halodiffuse stop\n  halodiffuse --prompt <TEXT> --path <FILE> [--seed <U64>]"
        );
        std::process::exit(1);
    }
    let sock_path = get_sock_path();
    let pid_path = get_pid_path();
    match args[0].as_str() {
        "__daemon" => {
            run_daemon();
        }
        "start" => {
            if UnixStream::connect(&sock_path).is_ok() {
                println!("halodiffuse is already running.");
                return;
            }
            let _ = fs::remove_file(&sock_path);
            let _ = fs::remove_file(&pid_path);
            let child = Command::new(env::current_exe().unwrap())
                .arg("__daemon")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap();
            println!(
                "Starting halodiffuse daemon (PID {}), loading weights...",
                child.id()
            );
            while !sock_path.exists() {
                std::thread::sleep(Duration::from_millis(100));
            }
            println!("halodiffuse started and ready.");
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
            } else if let Ok(pid_str) = fs::read_to_string(&pid_path) {
                if let Ok(pid) = pid_str.trim().parse::<i32>() {
                    unsafe {
                        unsafe extern "C" {
                            fn kill(pid: i32, sig: i32) -> i32;
                        }
                        if kill(pid, 15) == 0 {
                            println!("Stopped halodiffuse daemon (PID {pid}).");
                            stopped = true;
                        }
                    }
                }
            }
            let _ = fs::remove_file(&sock_path);
            let _ = fs::remove_file(&pid_path);
            if !stopped {
                eprintln!("halodiffuse is not running.");
                std::process::exit(1);
            }
        }
        _ => {
            let mut prompt: Option<String> = None;
            let mut out_path: Option<PathBuf> = None;
            let mut seed: Option<u64> = None;
            let mut i = 0;
            while i < args.len() {
                match args[i].as_str() {
                    "--prompt" | "—prompt" => {
                        i += 1;
                        prompt = args.get(i).cloned();
                    }
                    "--path" | "—path" => {
                        i += 1;
                        out_path = args.get(i).map(PathBuf::from);
                    }
                    "--seed" | "—seed" => {
                        i += 1;
                        seed = Some(
                            args.get(i)
                                .expect("Missing value for --seed")
                                .parse::<u64>()
                                .expect("Invalid u64 for --seed"),
                        );
                    }
                    other => {
                        eprintln!("Unknown argument: {other}");
                        std::process::exit(1);
                    }
                }
                i += 1;
            }
            let prompt = prompt.expect("Missing required --prompt <TEXT>");
            let out_path = out_path.expect("Missing required --path <FILE>");
            if out_path.is_dir() {
                eprintln!("Error: --path must be a file path, not a directory.");
                std::process::exit(1);
            }
            let abs_out_path = if out_path.is_absolute() {
                out_path
            } else {
                env::current_dir().unwrap().join(out_path)
            };
            let seed = seed.unwrap_or_else(read_urandom_u64);
            let mut stream = UnixStream::connect(&sock_path).unwrap_or_else(|_| {
                eprintln!("halodiffuse daemon is not running. Start it with: halodiffuse start");
                std::process::exit(1);
            });
            let prompt_bytes = prompt.as_bytes();
            let path_str = abs_out_path.to_str().unwrap();
            let path_bytes = path_str.as_bytes();
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
