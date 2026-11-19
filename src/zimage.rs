use crate::engine::{
    ArenaBump, BoundDispatch, DeviceScratchArena, Engine, EulerPushConstants, NoisePushConstants,
    WeightPack,
};
use crate::vae::{VaeWorkspace, record_vae_cmd};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{CopyBufferInfo, PrimaryAutoCommandBuffer};

#[repr(C)]
#[derive(Clone, Copy, BufferContents)]
struct CapPadPushConstants {
    valid_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, BufferContents)]
struct Rope3dPushConstants {
    t_offset: u32,
}

struct ZImageWorkspace {
    qwen_x: Subbuffer<[u8]>,
    qwen_norm: Subbuffer<[u8]>,
    qwen_qkv: Subbuffer<[u8]>,
    qwen_attn_out: Subbuffer<[u8]>,
    qwen_ffn_hid: Subbuffer<[u8]>,
    t_hid: Subbuffer<[u8]>,
    t_emb: Subbuffer<[u8]>,
    adaln_out: Subbuffer<[u8]>,
    unified: Subbuffer<[u8]>,
    unified_norm: Subbuffer<[u8]>,
    unified_qkv: Subbuffer<[u8]>,
    unified_attn: Subbuffer<[u8]>,
    unified_mlp: Subbuffer<[u8]>,
}

impl ZImageWorkspace {
    fn new(arena: &DeviceScratchArena) -> Self {
        let mut bump = ArenaBump::new(&arena.arena_a, arena.align);
        Self {
            qwen_x: bump.alloc(256 * 2560 * 2),
            qwen_norm: bump.alloc(256 * 2560 * 2),
            qwen_qkv: bump.alloc(256 * 6144 * 2),
            qwen_attn_out: bump.alloc(256 * 4096 * 2),
            qwen_ffn_hid: bump.alloc(256 * 9728 * 2),
            t_hid: bump.alloc(1024 * 2),
            t_emb: bump.alloc(256 * 2),
            adaln_out: bump.alloc(15360 * 2),
            unified: bump.alloc(4352 * 3840 * 2),
            unified_norm: bump.alloc(4352 * 3840 * 2),
            unified_qkv: bump.alloc(4352 * 11520 * 2),
            unified_attn: bump.alloc(4352 * 3840 * 2),
            unified_mlp: bump.alloc(4352 * 10240 * 2),
        }
    }
}

pub struct ZImagePipeline {
    engine: Engine,
    tok_qwen: Tokenizer,
    buf_tokens: Subbuffer<[u8]>,
    rgb_out: Subbuffer<[u8]>,
    num_latent_pairs: u32,
    noise_disp: BoundDispatch,
    qwen_cmd: Arc<PrimaryAutoCommandBuffer>,
    cap_pad_disp: BoundDispatch,
    refiner_cmd: Arc<PrimaryAutoCommandBuffer>,
    step_cmds: [Arc<PrimaryAutoCommandBuffer>; 8],
    vae_cmd: Arc<PrimaryAutoCommandBuffer>,
}

impl ZImagePipeline {
    pub fn load(data_dir: &Path) -> Self {
        let engine = Engine::new();
        let latents_x = engine.alloc_device_bytes(4096 * 64 * 2);
        let v_pred = engine.alloc_device_bytes(4096 * 64 * 2);
        let latents_nchw = engine.alloc_device_bytes(16 * 128 * 128 * 2);
        let rgb_out = engine.alloc_host_bytes(1024 * 1024 * 3);
        let arena = DeviceScratchArena::new(&engine);
        let vae_ws = VaeWorkspace::new(&engine, &arena);
        let tok_qwen = Tokenizer::from_bytes(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/tokenizer_qwen3.json"
        )))
        .unwrap();
        let staging = engine.alloc_staging_bytes(256 * 1024 * 1024);
        let w_qwen = WeightPack::load(&engine, &staging, data_dir.join("qwen3.safetensors"));
        let w_s3dit = WeightPack::load(&engine, &staging, data_dir.join("s3dit.safetensors"));
        let w_vae = WeightPack::load(&engine, &staging, data_dir.join("vae.safetensors"));
        drop(staging);
        let buf_tokens = engine.alloc_host_bytes(256 * 4);
        let txt_refined = engine.alloc_device_bytes(256 * 3840 * 2);
        let ws = ZImageWorkspace::new(&arena);
        let num_latent_pairs = (latents_x.size() / 4) as u32;
        let noise_disp = engine.bind(
            "gaussian_noise_splitmix64.comp",
            [num_latent_pairs.div_ceil(128), 1, 1],
            std::slice::from_ref(&latents_x),
        );
        let qwen_cmd = record_qwen_cmd(&engine, &ws, &w_qwen, &w_s3dit, &buf_tokens);
        let cap_pad_disp = engine.bind(
            "hgemm_m256_n3840_k2560_bias_cap_pad.comp",
            [30, 2, 1],
            &[
                ws.qwen_norm.clone(),
                w_s3dit.get("cap_embedder.linear.weight"),
                w_s3dit.get("cap_embedder.linear.bias"),
                w_s3dit.get("cap_pad_token"),
                txt_refined.clone(),
            ],
        );
        let refiner_cmd = record_context_refiner_cmd(&engine, &ws, &w_s3dit, &txt_refined);
        let mut sigmas = [0.0f32; 9];
        for (i, s_out) in sigmas.iter_mut().take(8).enumerate() {
            let s = 1.0f32 - (i as f32) / 8.0f32;
            *s_out = (3.0f32 * s) / (1.0f32 + 2.0f32 * s);
        }
        sigmas[8] = 0.0;
        let t_buffers: Vec<Subbuffer<[u8]>> = (0..8)
            .map(|step| {
                let t = (1.0f32 - sigmas[step]) * 1000.0f32;
                let b = engine.alloc_host_bytes(4);
                b.write().unwrap().copy_from_slice(&t.to_le_bytes());
                b
            })
            .collect();
        let step_cmds: [Arc<PrimaryAutoCommandBuffer>; 8] = std::array::from_fn(|step| {
            let dt = sigmas[step] - sigmas[step + 1];
            let euler_nchw = if step == 7 { Some(&latents_nchw) } else { None };
            record_s3dit_step_cmd(
                &engine,
                &ws,
                &w_s3dit,
                &txt_refined,
                &latents_x,
                &t_buffers[step],
                &v_pred,
                euler_nchw,
                dt,
            )
        });
        let vae_cmd = record_vae_cmd(&engine, &vae_ws, &w_vae, &latents_nchw, &rgb_out);
        Self {
            engine,
            tok_qwen,
            buf_tokens,
            rgb_out,
            num_latent_pairs,
            noise_disp,
            qwen_cmd,
            cap_pad_disp,
            refiner_cmd,
            step_cmds,
            vae_cmd,
        }
    }

    pub fn generate(&self, prompt: &str, seed: u64) -> Vec<u8> {
        let (tok_bytes, valid_len) = encode_qwen3_tokens(&self.tok_qwen, prompt);
        self.buf_tokens.write().unwrap().copy_from_slice(&tok_bytes);
        let mut cmd = self.engine.new_one_time_cmd();
        self.noise_disp.record_push(
            &mut cmd,
            NoisePushConstants {
                seed,
                num_pairs: self.num_latent_pairs,
                _pad: 0,
            },
        );
        self.engine.submit_and_wait(cmd);
        self.engine.submit_recorded_and_wait(&self.qwen_cmd);
        let mut cap_cmd = self.engine.new_one_time_cmd();
        self.cap_pad_disp
            .record_push(&mut cap_cmd, CapPadPushConstants { valid_len });
        self.engine.submit_and_wait(cap_cmd);
        self.engine.submit_recorded_and_wait(&self.refiner_cmd);
        for step_cmd in &self.step_cmds {
            self.engine.submit_recorded_and_wait(step_cmd);
        }
        self.engine.submit_recorded_and_wait(&self.vae_cmd);
        self.rgb_out.read().unwrap().to_vec()
    }
}

fn encode_qwen3_tokens(tokenizer: &Tokenizer, prompt: &str) -> (Vec<u8>, u32) {
    let formatted = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let enc = tokenizer.encode(formatted, false).unwrap();
    let raw_ids = enc.get_ids();
    let valid_len = raw_ids.len().min(256) as u32;
    let mut ids = Vec::with_capacity(256);
    ids.extend(raw_ids.iter().copied().take(256));
    ids.resize(256, 151643);
    let bytes = ids.into_iter().flat_map(u32::to_le_bytes).collect();
    (bytes, valid_len)
}

fn record_qwen_cmd(
    engine: &Engine,
    ws: &ZImageWorkspace,
    w_qwen: &WeightPack,
    w_s3dit: &WeightPack,
    buf_tokens: &Subbuffer<[u8]>,
) -> Arc<PrimaryAutoCommandBuffer> {
    let mut cmd = engine.new_reusable_cmd();
    engine
        .bind(
            "embed_lookup_v151936_m256_d2560.comp",
            [256, 1, 1],
            &[
                buf_tokens.clone(),
                w_qwen.get("embed_tokens.weight"),
                ws.qwen_x.clone(),
            ],
        )
        .record(&mut cmd);
    for i in 0..35 {
        engine
            .bind(
                "rmsnorm_m256_d2560.comp",
                [256, 1, 1],
                &[
                    ws.qwen_x.clone(),
                    w_qwen.get(&format!("layers.{i}.ln1.gamma")),
                    ws.qwen_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_n6144_k2560.comp",
                [48, 2, 1],
                &[
                    ws.qwen_norm.clone(),
                    w_qwen.get(&format!("layers.{i}.attn.qkv.weight")),
                    ws.qwen_qkv.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "qk_rmsnorm_rope1d_m256_h32_hkv8_d128.comp",
                [5, 256, 1],
                &[
                    ws.qwen_qkv.clone(),
                    w_qwen.get(&format!("layers.{i}.attn.q_norm")),
                    w_qwen.get(&format!("layers.{i}.attn.k_norm")),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "flash_attention_causal_gqa_m256_h32_hkv8_d128.comp",
                [16, 8, 1],
                &[ws.qwen_qkv.clone(), ws.qwen_attn_out.clone()],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_n2560_k4096_residual.comp",
                [20, 2, 1],
                &[
                    ws.qwen_attn_out.clone(),
                    w_qwen.get(&format!("layers.{i}.attn.o.weight")),
                    ws.qwen_x.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "rmsnorm_m256_d2560.comp",
                [256, 1, 1],
                &[
                    ws.qwen_x.clone(),
                    w_qwen.get(&format!("layers.{i}.ln2.gamma")),
                    ws.qwen_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_n9728_k2560_swiglu.comp",
                [152, 2, 1],
                &[
                    ws.qwen_norm.clone(),
                    w_qwen.get(&format!("layers.{i}.mlp.gate.weight")),
                    w_qwen.get(&format!("layers.{i}.mlp.up.weight")),
                    ws.qwen_ffn_hid.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_n2560_k9728_residual.comp",
                [20, 2, 1],
                &[
                    ws.qwen_ffn_hid.clone(),
                    w_qwen.get(&format!("layers.{i}.mlp.down.weight")),
                    ws.qwen_x.clone(),
                ],
            )
            .record(&mut cmd);
    }
    engine
        .bind(
            "rmsnorm_m256_d2560.comp",
            [256, 1, 1],
            &[
                ws.qwen_x.clone(),
                w_s3dit.get("cap_embedder.norm.gamma"),
                ws.qwen_norm.clone(),
            ],
        )
        .record(&mut cmd);
    cmd.build().unwrap()
}

fn record_context_refiner_cmd(
    engine: &Engine,
    ws: &ZImageWorkspace,
    w: &WeightPack,
    txt_refined: &Subbuffer<[u8]>,
) -> Arc<PrimaryAutoCommandBuffer> {
    let txt_norm = ws.unified_norm.clone().slice(0..(256 * 3840 * 2));
    let txt_qkv = ws.unified_qkv.clone().slice(0..(256 * 11520 * 2));
    let txt_attn = ws.unified_attn.clone().slice(0..(256 * 3840 * 2));
    let txt_mlp = ws.unified_mlp.clone().slice(0..(256 * 10240 * 2));
    let mut cmd = engine.new_reusable_cmd();
    for i in 0..2 {
        let p = format!("context_refiner.{i}");
        engine
            .bind(
                "rmsnorm_m256_d3840.comp",
                [256, 1, 1],
                &[
                    txt_refined.clone(),
                    w.get(&format!("{p}.attn_norm1")),
                    txt_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_4096_4352_n11520_k3840.comp",
                [90, 2, 1],
                &[
                    txt_norm.clone(),
                    w.get(&format!("{p}.attn.qkv.weight")),
                    txt_qkv.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "qk_rmsnorm_rope3d_m256_h30_d128.comp",
                [8, 256, 1],
                &[
                    txt_qkv.clone(),
                    w.get(&format!("{p}.attn.norm_q")),
                    w.get(&format!("{p}.attn.norm_k")),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "flash_attention_m256_h30_d128.comp",
                [4, 30, 1],
                &[txt_qkv.clone(), txt_attn.clone()],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_4096_4352_n3840_k3840.comp",
                [30, 2, 1],
                &[
                    txt_attn.clone(),
                    w.get(&format!("{p}.attn.out.weight")),
                    txt_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "rmsnorm_m256_d3840_residual.comp",
                [256, 1, 1],
                &[
                    txt_norm.clone(),
                    w.get(&format!("{p}.attn_norm2")),
                    txt_refined.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "rmsnorm_m256_d3840.comp",
                [256, 1, 1],
                &[
                    txt_refined.clone(),
                    w.get(&format!("{p}.ffn_norm1")),
                    txt_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_4096_4352_n10240_k3840_swiglu.comp",
                [160, 2, 1],
                &[
                    txt_norm.clone(),
                    w.get(&format!("{p}.ffn.w1.weight")),
                    w.get(&format!("{p}.ffn.w3.weight")),
                    txt_mlp.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_4096_4352_n3840_k10240.comp",
                [30, 2, 1],
                &[
                    txt_mlp.clone(),
                    w.get(&format!("{p}.ffn.w2.weight")),
                    txt_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "rmsnorm_m256_d3840_residual.comp",
                [256, 1, 1],
                &[
                    txt_norm.clone(),
                    w.get(&format!("{p}.ffn_norm2")),
                    txt_refined.clone(),
                ],
            )
            .record(&mut cmd);
    }
    cmd.build().unwrap()
}

#[allow(clippy::too_many_arguments)]
fn record_s3dit_step_cmd(
    engine: &Engine,
    ws: &ZImageWorkspace,
    w: &WeightPack,
    txt_refined: &Subbuffer<[u8]>,
    latents_x: &Subbuffer<[u8]>,
    t_buf: &Subbuffer<[u8]>,
    v_out: &Subbuffer<[u8]>,
    euler_out_nchw: Option<&Subbuffer<[u8]>>,
    dt: f32,
) -> Arc<PrimaryAutoCommandBuffer> {
    let d_bytes = 3840 * 2;
    let ada_chunk = |idx: u64| -> Subbuffer<[u8]> {
        ws.adaln_out
            .clone()
            .slice((idx * d_bytes)..((idx + 1) * d_bytes))
    };
    let img = ws.unified.clone().slice(0..(4096 * d_bytes));
    let txt_dst = ws.unified.clone().slice((4096 * d_bytes)..(4352 * d_bytes));
    let img_norm = ws.unified_norm.clone().slice(0..(4096 * d_bytes));
    let img_qkv = ws.unified_qkv.clone().slice(0..(4096 * 3 * d_bytes));
    let img_attn = ws.unified_attn.clone().slice(0..(4096 * d_bytes));
    let img_mlp = ws.unified_mlp.clone().slice(0..(4096 * 10240 * 2));
    let mut cmd = engine.new_reusable_cmd();
    cmd.copy_buffer(CopyBufferInfo::buffers(txt_refined.clone(), txt_dst))
        .unwrap();
    engine
        .bind(
            "timestep_embed_sinusoidal_hgemm_m1_n1024_k256_bias_silu.comp",
            [8, 1, 1],
            &[
                t_buf.clone(),
                w.get("t_embedder.mlp.0.weight"),
                w.get("t_embedder.mlp.0.bias"),
                ws.t_hid.clone(),
            ],
        )
        .record(&mut cmd);
    engine
        .bind(
            "hgemm_m1_n256_k1024_bias.comp",
            [2, 1, 1],
            &[
                ws.t_hid.clone(),
                w.get("t_embedder.mlp.2.weight"),
                w.get("t_embedder.mlp.2.bias"),
                ws.t_emb.clone(),
            ],
        )
        .record(&mut cmd);
    engine
        .bind(
            "patch_embed_conv2d_m4096_n3840_k64.comp",
            [30, 32, 1],
            &[
                latents_x.clone(),
                w.get("x_embedder.weight"),
                w.get("x_embedder.bias"),
                img.clone(),
            ],
        )
        .record(&mut cmd);
    for i in 0..2 {
        let p = format!("noise_refiner.{i}");
        engine
            .bind(
                "adaln_mod_m1_n15360_k256_bias_tanh.comp",
                [120, 1, 1],
                &[
                    ws.t_emb.clone(),
                    w.get(&format!("{p}.adaln.weight")),
                    w.get(&format!("{p}.adaln.bias")),
                    ws.adaln_out.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "rmsnorm_adaln_m4096_4352_d3840.comp",
                [4096, 1, 1],
                &[
                    img.clone(),
                    w.get(&format!("{p}.attn_norm1")),
                    ada_chunk(0),
                    img_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_4096_4352_n11520_k3840.comp",
                [90, 32, 1],
                &[
                    img_norm.clone(),
                    w.get(&format!("{p}.attn.qkv.weight")),
                    img_qkv.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "qk_rmsnorm_rope3d_m4096_h30_d128.comp",
                [8, 4096, 1],
                &[
                    img_qkv.clone(),
                    w.get(&format!("{p}.attn.norm_q")),
                    w.get(&format!("{p}.attn.norm_k")),
                ],
            )
            .record_push(&mut cmd, Rope3dPushConstants { t_offset: 257 });
        engine
            .bind(
                "flash_attention_m4096_h30_d128.comp",
                [64, 30, 1],
                &[img_qkv.clone(), img_attn.clone()],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_4096_4352_n3840_k3840.comp",
                [30, 32, 1],
                &[
                    img_attn.clone(),
                    w.get(&format!("{p}.attn.out.weight")),
                    img_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "rmsnorm_d3840_gated_residual_m4096_4352.comp",
                [4096, 1, 1],
                &[
                    img_norm.clone(),
                    w.get(&format!("{p}.attn_norm2")),
                    ada_chunk(1),
                    img.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "rmsnorm_adaln_m4096_4352_d3840.comp",
                [4096, 1, 1],
                &[
                    img.clone(),
                    w.get(&format!("{p}.ffn_norm1")),
                    ada_chunk(2),
                    img_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_4096_4352_n10240_k3840_swiglu.comp",
                [160, 32, 1],
                &[
                    img_norm.clone(),
                    w.get(&format!("{p}.ffn.w1.weight")),
                    w.get(&format!("{p}.ffn.w3.weight")),
                    img_mlp.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_4096_4352_n3840_k10240.comp",
                [30, 32, 1],
                &[
                    img_mlp.clone(),
                    w.get(&format!("{p}.ffn.w2.weight")),
                    img_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "rmsnorm_d3840_gated_residual_m4096_4352.comp",
                [4096, 1, 1],
                &[
                    img_norm.clone(),
                    w.get(&format!("{p}.ffn_norm2")),
                    ada_chunk(3),
                    img.clone(),
                ],
            )
            .record(&mut cmd);
    }
    for i in 0..30 {
        let p = format!("layers.{i}");
        engine
            .bind(
                "adaln_mod_m1_n15360_k256_bias_tanh.comp",
                [120, 1, 1],
                &[
                    ws.t_emb.clone(),
                    w.get(&format!("{p}.adaln.weight")),
                    w.get(&format!("{p}.adaln.bias")),
                    ws.adaln_out.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "rmsnorm_adaln_m4096_4352_d3840.comp",
                [4352, 1, 1],
                &[
                    ws.unified.clone(),
                    w.get(&format!("{p}.attn_norm1")),
                    ada_chunk(0),
                    ws.unified_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_4096_4352_n11520_k3840.comp",
                [90, 34, 1],
                &[
                    ws.unified_norm.clone(),
                    w.get(&format!("{p}.attn.qkv.weight")),
                    ws.unified_qkv.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "qk_rmsnorm_rope3d_m4352_h30_d128.comp",
                [8, 4352, 1],
                &[
                    ws.unified_qkv.clone(),
                    w.get(&format!("{p}.attn.norm_q")),
                    w.get(&format!("{p}.attn.norm_k")),
                ],
            )
            .record_push(&mut cmd, Rope3dPushConstants { t_offset: 257 });
        engine
            .bind(
                "flash_attention_m4352_h30_d128.comp",
                [68, 30, 1],
                &[ws.unified_qkv.clone(), ws.unified_attn.clone()],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_4096_4352_n3840_k3840.comp",
                [30, 34, 1],
                &[
                    ws.unified_attn.clone(),
                    w.get(&format!("{p}.attn.out.weight")),
                    ws.unified_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "rmsnorm_d3840_gated_residual_m4096_4352.comp",
                [4352, 1, 1],
                &[
                    ws.unified_norm.clone(),
                    w.get(&format!("{p}.attn_norm2")),
                    ada_chunk(1),
                    ws.unified.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "rmsnorm_adaln_m4096_4352_d3840.comp",
                [4352, 1, 1],
                &[
                    ws.unified.clone(),
                    w.get(&format!("{p}.ffn_norm1")),
                    ada_chunk(2),
                    ws.unified_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_4096_4352_n10240_k3840_swiglu.comp",
                [160, 34, 1],
                &[
                    ws.unified_norm.clone(),
                    w.get(&format!("{p}.ffn.w1.weight")),
                    w.get(&format!("{p}.ffn.w3.weight")),
                    ws.unified_mlp.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "hgemm_m256_4096_4352_n3840_k10240.comp",
                [30, 34, 1],
                &[
                    ws.unified_mlp.clone(),
                    w.get(&format!("{p}.ffn.w2.weight")),
                    ws.unified_norm.clone(),
                ],
            )
            .record(&mut cmd);
        engine
            .bind(
                "rmsnorm_d3840_gated_residual_m4096_4352.comp",
                [4352, 1, 1],
                &[
                    ws.unified_norm.clone(),
                    w.get(&format!("{p}.ffn_norm2")),
                    ada_chunk(3),
                    ws.unified.clone(),
                ],
            )
            .record(&mut cmd);
    }
    engine
        .bind(
            "adaln_mod_m1_n3840_k256_silu_bias.comp",
            [30, 1, 1],
            &[
                ws.t_emb.clone(),
                w.get("final_layer.adaln.weight"),
                w.get("final_layer.adaln.bias"),
                ada_chunk(0),
            ],
        )
        .record(&mut cmd);
    engine
        .bind(
            "layernorm_adaln_scale_m4096_d3840.comp",
            [4096, 1, 1],
            &[img, ada_chunk(0), img_norm.clone()],
        )
        .record(&mut cmd);
    engine
        .bind(
            "hgemm_m4096_n64_k3840_bias.comp",
            [32, 1, 1],
            &[
                img_norm,
                w.get("final_layer.linear.weight"),
                w.get("final_layer.linear.bias"),
                v_out.clone(),
            ],
        )
        .record(&mut cmd);
    if let Some(nchw_out) = euler_out_nchw {
        engine
            .bind(
                "unpatchify_m4096_k64_to_latents_c16_h128_w128_euler_scale_flux.comp",
                [256, 1, 1],
                &[v_out.clone(), latents_x.clone(), nchw_out.clone()],
            )
            .record_push(&mut cmd, EulerPushConstants { dt });
    } else {
        engine
            .bind(
                "flow_match_euler_step_m4096_k64.comp",
                [1024, 1, 1],
                &[v_out.clone(), latents_x.clone()],
            )
            .record_push(&mut cmd, EulerPushConstants { dt });
    }
    cmd.build().unwrap()
}
