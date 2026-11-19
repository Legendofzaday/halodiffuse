use crate::engine::{
    ArenaBump, BoundDispatch, DeviceScratchArena, Engine, EulerPushConstants, NoisePushConstants,
    WeightPack,
};
use crate::vae::{VaeWorkspace, record_vae_cmd};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use vulkano::DeviceSize;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, PrimaryAutoCommandBuffer};

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

pub struct Sd35Pipeline {
    engine: Engine,
    tok_l: Tokenizer,
    tok_g: Tokenizer,
    tok_t5: Tokenizer,
    buf_tokens_l: Subbuffer<[u8]>,
    buf_tokens_g: Subbuffer<[u8]>,
    buf_tokens_t5: Subbuffer<[u8]>,
    rgb_out: Subbuffer<[u8]>,
    num_latent_pairs: u32,
    prebuilt_text: PrebuiltTextEncoders,
    mmdit_cmds: [Arc<PrimaryAutoCommandBuffer>; 4],
    vae_cmd: Arc<PrimaryAutoCommandBuffer>,
}

impl Sd35Pipeline {
    pub fn load(data_dir: &Path) -> Self {
        let engine = Engine::new();
        let latents_x = engine.alloc_device_bytes(4096 * 64 * 2);
        let v_pred = engine.alloc_device_bytes(4096 * 64 * 2);
        let latents_nchw = engine.alloc_device_bytes(16 * 128 * 128 * 2);
        let rgb_out = engine.alloc_host_bytes(1024 * 1024 * 3);
        let arena = DeviceScratchArena::new(&engine);
        let vae_ws = VaeWorkspace::new(&engine, &arena);
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
        let staging = engine.alloc_staging_bytes(256 * 1024 * 1024);
        let w_clip_l = WeightPack::load(&engine, &staging, data_dir.join("clip_l.safetensors"));
        let w_clip_g = WeightPack::load(&engine, &staging, data_dir.join("clip_g.safetensors"));
        let w_t5 = WeightPack::load(&engine, &staging, data_dir.join("t5xxl.safetensors"));
        let w_mmdit = WeightPack::load(&engine, &staging, data_dir.join("mmdit.safetensors"));
        let w_vae = WeightPack::load(&engine, &staging, data_dir.join("vae.safetensors"));
        drop(staging);
        let buf_tokens_l = engine.alloc_host_bytes(77 * 4);
        let buf_tokens_g = engine.alloc_host_bytes(77 * 4);
        let buf_tokens_t5 = engine.alloc_host_bytes(256 * 4);
        let clip_l_seq = engine.alloc_device_bytes(77 * 768 * 2);
        let clip_g_seq = engine.alloc_device_bytes(77 * 1280 * 2);
        let y_pooled = engine.alloc_device_bytes(2048 * 2);
        let ctx_333 = engine.alloc_device_bytes(333 * 4096 * 2);
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
        let text_ws = TextEncoderWorkspace::new(&arena);
        let mmdit_ws = MmditWorkspace::new(&arena);
        let prebuilt_text = build_text_encoders(
            &engine,
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
            record_mmdit_step_cmd(
                &engine,
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
        let vae_cmd = record_vae_cmd(&engine, &vae_ws, &w_vae, &latents_nchw, &rgb_out);
        let num_latent_pairs = (latents_x.size() / 4) as u32;
        Self {
            engine,
            tok_l,
            tok_g,
            tok_t5,
            buf_tokens_l,
            buf_tokens_g,
            buf_tokens_t5,
            rgb_out,
            num_latent_pairs,
            prebuilt_text,
            mmdit_cmds,
            vae_cmd,
        }
    }

    pub fn generate(&self, prompt: &str, seed: u64) -> Vec<u8> {
        let (clip_l_bytes, eos_l) = encode_clip_tokens(&self.tok_l, prompt, 49407);
        let (clip_g_bytes, eos_g) = encode_clip_tokens(&self.tok_g, prompt, 0);
        let t5_bytes = encode_t5_tokens(&self.tok_t5, prompt);
        self.buf_tokens_l
            .write()
            .unwrap()
            .copy_from_slice(&clip_l_bytes);
        self.buf_tokens_g
            .write()
            .unwrap()
            .copy_from_slice(&clip_g_bytes);
        self.buf_tokens_t5
            .write()
            .unwrap()
            .copy_from_slice(&t5_bytes);
        let mut cmd = self.engine.new_one_time_cmd();
        self.prebuilt_text.noise.record_push(
            &mut cmd,
            NoisePushConstants {
                seed,
                num_pairs: self.num_latent_pairs,
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
        record_clip(&mut cmd, &self.prebuilt_text.clip_l, eos_l);
        record_clip(&mut cmd, &self.prebuilt_text.clip_g, eos_g);
        self.prebuilt_text.clip_concat.record(&mut cmd);
        self.prebuilt_text.t5_emb.record(&mut cmd);
        for layer in &self.prebuilt_text.t5_layers {
            for disp in layer {
                disp.record(&mut cmd);
            }
        }
        self.prebuilt_text.t5_final_ln.record(&mut cmd);
        self.engine.submit_and_wait(cmd);
        for step_cmd in &self.mmdit_cmds {
            self.engine.submit_recorded_and_wait(step_cmd);
        }
        self.engine.submit_recorded_and_wait(&self.vae_cmd);
        self.rgb_out.read().unwrap().to_vec()
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

#[allow(clippy::too_many_arguments)]
fn build_clip(
    engine: &Engine,
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
    let emb = engine.bind(
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
            engine.bind(
                sh_ln,
                [77, 1, 1],
                &[
                    scratch.x.clone(),
                    w.get(&format!("layers.{i}.ln1.gamma")),
                    w.get(&format!("layers.{i}.ln1.beta")),
                    scratch.norm.clone(),
                ],
            ),
            engine.bind(
                sh_qkv,
                [nx_d * 3, 3, 1],
                &[
                    scratch.norm.clone(),
                    w.get(&format!("layers.{i}.attn.qkv.weight")),
                    w.get(&format!("layers.{i}.attn.qkv.bias")),
                    scratch.qkv.clone(),
                ],
            ),
            engine.bind(
                sh_fa,
                [1, heads, 1],
                &[scratch.qkv.clone(), scratch.attn_out.clone()],
            ),
            engine.bind(
                sh_out,
                [nx_d, 3, 1],
                &[
                    scratch.attn_out.clone(),
                    w.get(&format!("layers.{i}.attn.out_proj.weight")),
                    w.get(&format!("layers.{i}.attn.out_proj.bias")),
                    scratch.x.clone(),
                ],
            ),
            engine.bind(
                sh_ln,
                [77, 1, 1],
                &[
                    scratch.x.clone(),
                    w.get(&format!("layers.{i}.ln2.gamma")),
                    w.get(&format!("layers.{i}.ln2.beta")),
                    scratch.norm.clone(),
                ],
            ),
            engine.bind(
                sh_fc1,
                [nx_d * 4, 3, 1],
                &[
                    scratch.norm.clone(),
                    w.get(&format!("layers.{i}.mlp.fc1.weight")),
                    w.get(&format!("layers.{i}.mlp.fc1.bias")),
                    scratch.mlp_hid.clone(),
                ],
            ),
            engine.bind(
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
            engine.bind(
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

#[allow(clippy::too_many_arguments)]
fn build_text_encoders(
    engine: &Engine,
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
    let noise = engine.bind(
        "gaussian_noise_splitmix64.comp",
        [num_pairs.div_ceil(128), 1, 1],
        std::slice::from_ref(latents_x),
    );
    let y_l_slice = y_pooled.clone().slice(0..(768 * 2));
    let y_g_slice = y_pooled.clone().slice((768 * 2)..(2048 * 2));
    let clip_l = build_clip(
        engine,
        w_clip_l,
        &ws.clip_l,
        buf_tokens_l,
        768,
        12,
        clip_l_seq,
        &y_l_slice,
    );
    let clip_g = build_clip(
        engine,
        w_clip_g,
        &ws.clip_g,
        buf_tokens_g,
        1280,
        32,
        clip_g_seq,
        &y_g_slice,
    );
    let ctx_clip_slice = ctx_333.clone().slice(0..(77 * 4096 * 2));
    let clip_concat = engine.bind(
        "clip_concat_pad_m77_d4096.comp",
        [77, 1, 1],
        &[clip_l_seq.clone(), clip_g_seq.clone(), ctx_clip_slice],
    );
    let rpb = w_t5.get("rpb_weight");
    let t5_emb = engine.bind(
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
            engine.bind(
                "rmsnorm_m256_d4096.comp",
                [256, 1, 1],
                &[
                    ws.t5_x.clone(),
                    w_t5.get(&format!("blocks.{i}.ln1.gamma")),
                    ws.t5_norm.clone(),
                ],
            ),
            engine.bind(
                "hgemm_m256_n12288_k4096.comp",
                [96, 2, 1],
                &[
                    ws.t5_norm.clone(),
                    w_t5.get(&format!("blocks.{i}.attn.qkv.weight")),
                    ws.t5_qkv.clone(),
                ],
            ),
            engine.bind(
                "flash_attention_rpb_m256_h64_d64.comp",
                [4, 64, 1],
                &[ws.t5_qkv.clone(), rpb.clone(), ws.t5_attn_out.clone()],
            ),
            engine.bind(
                "hgemm_m256_n4096_k4096_residual.comp",
                [32, 2, 1],
                &[
                    ws.t5_attn_out.clone(),
                    w_t5.get(&format!("blocks.{i}.attn.o.weight")),
                    ws.t5_x.clone(),
                ],
            ),
            engine.bind(
                "rmsnorm_m256_d4096.comp",
                [256, 1, 1],
                &[
                    ws.t5_x.clone(),
                    w_t5.get(&format!("blocks.{i}.ln2.gamma")),
                    ws.t5_norm.clone(),
                ],
            ),
            engine.bind(
                "hgemm_m256_n10240_k4096_geglu.comp",
                [160, 2, 1],
                &[
                    ws.t5_norm.clone(),
                    w_t5.get(&format!("blocks.{i}.ffn.b0_wi0.weight")),
                    w_t5.get(&format!("blocks.{i}.ffn.b1_wi1.weight")),
                    ws.t5_ffn_hid.clone(),
                ],
            ),
            engine.bind(
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
    let t5_final_ln = engine.bind(
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

#[allow(clippy::too_many_arguments)]
fn record_mmdit_step_cmd(
    engine: &Engine,
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
    let mut cmd = engine.new_reusable_cmd();
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
            engine
                .bind(
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
            engine
                .bind(
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
            engine
                .bind(
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
        engine
            .bind(
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
        engine
            .bind(
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
        engine
            .bind(
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
        engine
            .bind(
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
        engine
            .bind(
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
        engine
            .bind(
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
        engine
            .bind(
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
            engine
                .bind(
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
        engine
            .bind(
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
            engine
                .bind(
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
        engine
            .bind(
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
            engine
                .bind(
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
        engine
            .bind(
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
            engine
                .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
        engine
            .bind(
                "unpatchify_m4096_k64_to_latents_c16_h128_w128_euler_scale.comp",
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
