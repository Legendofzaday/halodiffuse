use hf_hub::HFClientSync;
use rayon::prelude::*;
use safetensors::Dtype;
use safetensors::tensor::{SafeTensors, TensorView, serialize_to_file};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{create_dir_all, read, read_to_string, rename};
use std::io::{Write, stdout};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Instant;

const SD35_OWNER: &str = "stabilityai";
const SD35_NAME: &str = "stable-diffusion-3.5-large-turbo";
const ZIMAGE_OWNER: &str = "Tongyi-MAI";
const ZIMAGE_NAME: &str = "Z-Image-Turbo";
const PERM_N: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

type PackedMap = BTreeMap<String, (Vec<usize>, Vec<u8>)>;

struct RawFile<'a> {
    st: SafeTensors<'a>,
}

impl<'a> RawFile<'a> {
    fn from_bytes(bytes: &'a [u8]) -> Self {
        Self {
            st: SafeTensors::deserialize(bytes).unwrap(),
        }
    }

    fn has(&self, key: &str) -> bool {
        self.st.tensor(key).is_ok()
    }

    fn read_f32(&self, key: &str) -> (Vec<usize>, Vec<f32>) {
        decode_view(key, &self.st.tensor(key).unwrap())
    }
}

struct RawSet<'a> {
    shards: Vec<SafeTensors<'a>>,
}

impl<'a> RawSet<'a> {
    fn from_buffers(buffers: &'a [Vec<u8>]) -> Self {
        Self {
            shards: buffers
                .iter()
                .map(|b| SafeTensors::deserialize(b).unwrap())
                .collect(),
        }
    }

    fn has(&self, key: &str) -> bool {
        self.shards.iter().any(|st| st.tensor(key).is_ok())
    }

    fn read_f32(&self, key: &str) -> (Vec<usize>, Vec<f32>) {
        let view = self
            .shards
            .iter()
            .find_map(|st| st.tensor(key).ok())
            .unwrap_or_else(|| panic!("missing tensor: {key}"));
        decode_view(key, &view)
    }
}

fn decode_view(key: &str, view: &TensorView<'_>) -> (Vec<usize>, Vec<f32>) {
    let shape = view.shape().to_vec();
    let raw = view.data();
    let vals = match view.dtype() {
        Dtype::F32 => raw
            .par_chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => raw
            .par_chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        Dtype::BF16 => raw
            .par_chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        other => panic!("unsupported dtype {other:?} for {key}"),
    };
    (shape, vals)
}

fn f32_to_f16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; vals.len() * 2];
    out.par_chunks_exact_mut(2)
        .zip(vals.par_iter())
        .for_each(|(dst, &v)| {
            let bytes = (v.clamp(-65504.0, 65504.0) as f16).to_le_bytes();
            dst[0] = bytes[0];
            dst[1] = bytes[1];
        });
    out
}

fn insert_f16(out: &mut PackedMap, key: impl Into<String>, shape: Vec<usize>, vals: &[f32]) {
    out.insert(key.into(), (shape, f32_to_f16_bytes(vals)));
}

fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; rows * cols];
    dst.par_chunks_exact_mut(rows)
        .enumerate()
        .for_each(|(c, dst_row)| {
            for r in 0..rows {
                dst_row[r] = src[r * cols + c];
            }
        });
    dst
}

fn pack_b_tiles_16x16(b_logical: &[f32], k: usize, n: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(16) && n.is_multiple_of(16));
    let nt_max = n / 16;
    let mut packed = vec![0u8; k * n * 2];
    let kt_stride = 16 * nt_max * 16 * 2;
    packed
        .par_chunks_exact_mut(kt_stride)
        .enumerate()
        .for_each(|(kt, kt_slice)| {
            for (n_in, &nc) in PERM_N.iter().enumerate() {
                for nt in 0..nt_max {
                    let col = nt * 16 + nc;
                    let dst_base = (n_in * nt_max + nt) * 16 * 2;
                    for kr in 0..16 {
                        let row = kt * 16 + kr;
                        let v = b_logical[row * n + col].clamp(-65504.0, 65504.0);
                        let bytes = (v as f16).to_le_bytes();
                        kt_slice[dst_base + kr * 2] = bytes[0];
                        kt_slice[dst_base + kr * 2 + 1] = bytes[1];
                    }
                }
            }
        });
    packed
}

fn insert_packed_nk(out: &mut PackedMap, key: impl Into<String>, w_nk: &[f32], k: usize, n: usize) {
    let w_kn = transpose_2d(w_nk, n, k);
    out.insert(key.into(), (vec![k, n], pack_b_tiles_16x16(&w_kn, k, n)));
}

fn save_packed(out: &PackedMap, path: &Path) {
    let tmp = path.with_extension("safetensors.tmp");
    let views: Vec<(&str, TensorView<'_>)> = out
        .iter()
        .map(|(k, (shape, bytes))| {
            (
                k.as_str(),
                TensorView::new(Dtype::F16, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    serialize_to_file(views, None, &tmp).unwrap();
    rename(&tmp, path).unwrap();
}

fn find_key(raw: &RawFile<'_>, prefixes: &[&str], suffix: &str) -> String {
    prefixes
        .iter()
        .map(|p| format!("{p}{suffix}"))
        .find(|cand| raw.has(cand))
        .unwrap_or_else(|| panic!("missing tensor: {suffix}"))
}

fn find_key_set(raw: &RawSet<'_>, prefixes: &[&str], suffix: &str) -> String {
    prefixes
        .iter()
        .map(|p| format!("{p}{suffix}"))
        .find(|cand| raw.has(cand))
        .unwrap_or_else(|| panic!("missing tensor: {suffix}"))
}

fn convert_clip(input_path: &Path, output_path: &Path, d_model: usize, num_layers: usize) {
    let raw_bytes = read(input_path).unwrap();
    let raw = RawFile::from_bytes(&raw_bytes);
    let pfx = [
        "text_encoders.clip_l.transformer.",
        "text_encoders.clip_g.transformer.",
        "",
    ];
    let mut out = PackedMap::new();
    for (dst_k, src_k) in [
        (
            "embeddings.token_embedding.weight",
            "text_model.embeddings.token_embedding.weight",
        ),
        (
            "embeddings.position_embedding.weight",
            "text_model.embeddings.position_embedding.weight",
        ),
    ] {
        let (shape, vals) = raw.read_f32(&find_key(&raw, &pfx, src_k));
        insert_f16(&mut out, dst_k, shape, &vals);
    }
    for i in 0..num_layers {
        let p = format!("text_model.encoder.layers.{i}");
        for (dst_s, src_s) in [
            ("ln1.gamma", "layer_norm1.weight"),
            ("ln1.beta", "layer_norm1.bias"),
            ("ln2.gamma", "layer_norm2.weight"),
            ("ln2.beta", "layer_norm2.bias"),
        ] {
            let (shape, vals) = raw.read_f32(&find_key(&raw, &pfx, &format!("{p}.{src_s}")));
            insert_f16(&mut out, format!("layers.{i}.{dst_s}"), shape, &vals);
        }
        let (_, wq) = raw.read_f32(&find_key(
            &raw,
            &pfx,
            &format!("{p}.self_attn.q_proj.weight"),
        ));
        let (_, wk) = raw.read_f32(&find_key(
            &raw,
            &pfx,
            &format!("{p}.self_attn.k_proj.weight"),
        ));
        let (_, wv) = raw.read_f32(&find_key(
            &raw,
            &pfx,
            &format!("{p}.self_attn.v_proj.weight"),
        ));
        let mut wqkv = Vec::with_capacity(3 * d_model * d_model);
        wqkv.extend_from_slice(&wq);
        wqkv.extend_from_slice(&wk);
        wqkv.extend_from_slice(&wv);
        insert_packed_nk(
            &mut out,
            format!("layers.{i}.attn.qkv.weight"),
            &wqkv,
            d_model,
            3 * d_model,
        );
        let (_, bq) = raw.read_f32(&find_key(&raw, &pfx, &format!("{p}.self_attn.q_proj.bias")));
        let (_, bk) = raw.read_f32(&find_key(&raw, &pfx, &format!("{p}.self_attn.k_proj.bias")));
        let (_, bv) = raw.read_f32(&find_key(&raw, &pfx, &format!("{p}.self_attn.v_proj.bias")));
        let mut bqkv = Vec::with_capacity(3 * d_model);
        bqkv.extend_from_slice(&bq);
        bqkv.extend_from_slice(&bk);
        bqkv.extend_from_slice(&bv);
        insert_f16(
            &mut out,
            format!("layers.{i}.attn.qkv.bias"),
            vec![3 * d_model],
            &bqkv,
        );
        for (dst_s, src_s, k, n) in [
            ("attn.out_proj", "self_attn.out_proj", d_model, d_model),
            ("mlp.fc1", "mlp.fc1", d_model, 4 * d_model),
            ("mlp.fc2", "mlp.fc2", 4 * d_model, d_model),
        ] {
            let (_, w) = raw.read_f32(&find_key(&raw, &pfx, &format!("{p}.{src_s}.weight")));
            let (_, b) = raw.read_f32(&find_key(&raw, &pfx, &format!("{p}.{src_s}.bias")));
            insert_packed_nk(&mut out, format!("layers.{i}.{dst_s}.weight"), &w, k, n);
            insert_f16(&mut out, format!("layers.{i}.{dst_s}.bias"), vec![n], &b);
        }
    }
    let (s_g, fln_g) = raw.read_f32(&find_key(&raw, &pfx, "text_model.final_layer_norm.weight"));
    let (s_b, fln_b) = raw.read_f32(&find_key(&raw, &pfx, "text_model.final_layer_norm.bias"));
    let proj_w = if let Some(k) = pfx
        .iter()
        .map(|p| format!("{p}text_projection.weight"))
        .find(|k| raw.has(k))
    {
        transpose_2d(&raw.read_f32(&k).1, d_model, d_model)
    } else if let Some(k) = pfx
        .iter()
        .map(|p| format!("{p}text_projection"))
        .find(|k| raw.has(k))
    {
        raw.read_f32(&k).1
    } else {
        let mut eye = vec![0.0f32; d_model * d_model];
        for i in 0..d_model {
            eye[i * d_model + i] = 1.0;
        }
        eye
    };
    let proj_b = pfx
        .iter()
        .map(|p| format!("{p}text_projection.bias"))
        .find(|k| raw.has(k))
        .map(|k| raw.read_f32(&k).1)
        .unwrap_or_else(|| vec![0.0f32; d_model]);
    insert_f16(&mut out, "final_ln.gamma", s_g, &fln_g);
    insert_f16(&mut out, "final_ln.beta", s_b, &fln_b);
    insert_f16(&mut out, "eos_proj.weight", vec![d_model, d_model], &proj_w);
    insert_f16(&mut out, "eos_proj.bias", vec![d_model], &proj_b);
    save_packed(&out, output_path);
}

fn rmsnorm_fp16(x: &[f16], gamma: &[f16], rows: usize, cols: usize) -> Vec<f16> {
    let mut out = vec![0.0f16; rows * cols];
    out.par_chunks_exact_mut(cols)
        .zip(x.par_chunks_exact(cols))
        .for_each(|(out_row, row)| {
            let sq_sum: f32 = row.iter().map(|&v| (v as f32) * (v as f32)).sum();
            let rstd = 1.0 / (sq_sum / cols as f32 + 1e-6).sqrt();
            for c in 0..cols {
                out_row[c] = ((row[c] as f32) * rstd * (gamma[c] as f32)) as f16;
            }
        });
    out
}

fn matmul_nt_f16_in(a: &[f16], b_nk: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_row = &b_nk[j * k..(j + 1) * k];
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += (a_row[l] as f32) * b_row[l];
            }
            out_row[j] = sum;
        }
    });
    out
}

fn rms_gain(w_nk: &[f32], n: usize) -> f32 {
    let sum_sq: f64 = w_nk.par_iter().map(|&v| (v as f64) * (v as f64)).sum();
    (sum_sq / n as f64).sqrt() as f32
}

fn max_row_l2(w_nk: &[f32], n: usize, k: usize) -> f32 {
    w_nk.par_chunks_exact(k)
        .take(n)
        .map(|row| row.iter().map(|&v| v * v).sum::<f32>().sqrt())
        .reduce(|| 0.0f32, f32::max)
}

fn pow2_safe_div(est_rms: f32, est_peak: f32, rms_cap: f32, peak_cap: f32) -> f32 {
    let mut div = 1.0f32;
    while est_rms / div > rms_cap || (est_peak / div > peak_cap && est_rms / (div * 2.0) >= 0.05) {
        div *= 2.0;
    }
    div
}

fn qwen3_causal_gqa_f32(
    q_raw: &[f32],
    k_raw: &[f32],
    v_raw: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
) -> Vec<f16> {
    const SEQ: usize = 256;
    const HQ: usize = 32;
    const HKV: usize = 8;
    const D: usize = 128;
    let mut inv_freq = [0.0f32; D / 2];
    for d in 0..(D / 2) {
        inv_freq[d] = 1.0 / 1_000_000.0f32.powf((2 * d) as f32 / D as f32);
    }
    let mut q = vec![0.0f32; SEQ * HQ * D];
    for t in 0..SEQ {
        for h in 0..HQ {
            let src = &q_raw[(t * HQ + h) * D..(t * HQ + h + 1) * D];
            let sq: f32 = src.iter().map(|&v| v * v).sum();
            let rstd = 1.0 / (sq / D as f32 + 1e-6).sqrt();
            let dst = &mut q[(t * HQ + h) * D..(t * HQ + h + 1) * D];
            for d in 0..(D / 2) {
                let x0 = src[d] * rstd * q_norm[d];
                let x1 = src[d + D / 2] * rstd * q_norm[d + D / 2];
                let ang = (t as f32) * inv_freq[d];
                let (s, c) = ang.sin_cos();
                dst[d] = x0 * c - x1 * s;
                dst[d + D / 2] = x0 * s + x1 * c;
            }
        }
    }
    let mut k = vec![0.0f32; SEQ * HKV * D];
    for t in 0..SEQ {
        for h in 0..HKV {
            let src = &k_raw[(t * HKV + h) * D..(t * HKV + h + 1) * D];
            let sq: f32 = src.iter().map(|&v| v * v).sum();
            let rstd = 1.0 / (sq / D as f32 + 1e-6).sqrt();
            let dst = &mut k[(t * HKV + h) * D..(t * HKV + h + 1) * D];
            for d in 0..(D / 2) {
                let x0 = src[d] * rstd * k_norm[d];
                let x1 = src[d + D / 2] * rstd * k_norm[d + D / 2];
                let ang = (t as f32) * inv_freq[d];
                let (s, c) = ang.sin_cos();
                dst[d] = x0 * c - x1 * s;
                dst[d + D / 2] = x0 * s + x1 * c;
            }
        }
    }
    let scale = 1.0 / (D as f32).sqrt();
    let mut out = vec![0.0f32; SEQ * HQ * D];
    out.par_chunks_exact_mut(HQ * D)
        .enumerate()
        .for_each(|(tq, out_tok)| {
            let mut probs = [0.0f32; SEQ];
            for h in 0..HQ {
                let kvh = h / 4;
                let q_vec = &q[(tq * HQ + h) * D..(tq * HQ + h + 1) * D];
                let mut max_s = f32::NEG_INFINITY;
                for tk in 0..=tq {
                    let k_vec = &k[(tk * HKV + kvh) * D..(tk * HKV + kvh + 1) * D];
                    let mut dot = 0.0f32;
                    for d in 0..D {
                        dot += q_vec[d] * k_vec[d];
                    }
                    let s = dot * scale;
                    probs[tk] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                let mut denom = 0.0f32;
                for tk in 0..=tq {
                    let e = (probs[tk] - max_s).exp();
                    probs[tk] = e;
                    denom += e;
                }
                let inv_denom = 1.0 / denom;
                let out_head = &mut out_tok[h * D..(h + 1) * D];
                for tk in 0..=tq {
                    let p = probs[tk] * inv_denom;
                    let v_vec = &v_raw[(tk * HKV + kvh) * D..(tk * HKV + kvh + 1) * D];
                    for d in 0..D {
                        out_head[d] += p * v_vec[d];
                    }
                }
                for d in 0..D {
                    out_head[d] = (out_head[d].clamp(-65504.0, 65504.0) as f16) as f32;
                }
            }
        });
    out.into_iter().map(|v| v as f16).collect()
}

fn convert_t5(input_path: &Path, output_path: &Path) {
    let raw_bytes = read(input_path).unwrap();
    let raw = RawFile::from_bytes(&raw_bytes);
    let mut out = PackedMap::new();
    let emb_key = if raw.has("shared.weight") {
        "shared.weight"
    } else {
        "encoder.embed_tokens.weight"
    };
    let rpb_key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight";
    let res_scale = 32.0f32;
    let (emb_shape, mut emb_vals) = raw.read_f32(emb_key);
    let (rpb_shape, rpb_vals) = raw.read_f32(rpb_key);
    for v in &mut emb_vals {
        *v = (*v / res_scale) as f16 as f32;
    }
    insert_f16(&mut out, "embed_tokens.weight", emb_shape, &emb_vals);
    insert_f16(&mut out, "rpb_weight", rpb_shape, &rpb_vals);
    let mut x = vec![0.0f16; 256 * 4096];
    for c in 0..4096 {
        x[c] = emb_vals[4096 + c] as f16;
    }
    for r in 1..256 {
        for c in 0..4096 {
            x[r * 4096 + c] = emb_vals[c] as f16;
        }
    }
    for i in 0..24 {
        let p = format!("encoder.block.{i}");
        let (_, mut ln1_g) = raw.read_f32(&format!("{p}.layer.0.layer_norm.weight"));
        let (_, mut wq) = raw.read_f32(&format!("{p}.layer.0.SelfAttention.q.weight"));
        let (_, mut wk) = raw.read_f32(&format!("{p}.layer.0.SelfAttention.k.weight"));
        let (_, mut wv) = raw.read_f32(&format!("{p}.layer.0.SelfAttention.v.weight"));
        let (_, mut wo_attn) = raw.read_f32(&format!("{p}.layer.0.SelfAttention.o.weight"));
        for v in &mut wo_attn {
            *v /= res_scale;
        }
        let ln1_test_g: Vec<f16> = ln1_g.iter().map(|&v| v as f16).collect();
        let norm1_test = rmsnorm_fp16(&x, &ln1_test_g, 256, 4096);
        if norm1_test
            .iter()
            .any(|&v| v.is_infinite() || (v as f32).abs() > 8000.0)
        {
            for v in &mut ln1_g {
                *v /= 8.0;
            }
            for v in &mut wq {
                *v *= 8.0;
            }
            for v in &mut wk {
                *v *= 8.0;
            }
            for v in &mut wv {
                *v *= 8.0;
            }
        }
        let ln1_g_f16: Vec<f16> = ln1_g
            .iter()
            .map(|&v| v.clamp(-65504.0, 65504.0) as f16)
            .collect();
        let norm1 = rmsnorm_fp16(&x, &ln1_g_f16, 256, 4096);
        let v_f32 = matmul_nt_f16_in(&norm1, &wv, 256, 4096, 4096);
        if v_f32.iter().fold(0.0f32, |m, &v| m.max(v.abs())) > 16000.0 {
            for v in &mut wv {
                *v /= 8.0;
            }
            for v in &mut wo_attn {
                *v *= 8.0;
            }
        }
        insert_f16(
            &mut out,
            format!("blocks.{i}.ln1.gamma"),
            vec![4096],
            &ln1_g,
        );
        let mut qkv_nk = Vec::with_capacity(12288 * 4096);
        qkv_nk.extend_from_slice(&wq);
        qkv_nk.extend_from_slice(&wk);
        qkv_nk.extend_from_slice(&wv);
        insert_packed_nk(
            &mut out,
            format!("blocks.{i}.attn.qkv.weight"),
            &qkv_nk,
            4096,
            12288,
        );
        insert_packed_nk(
            &mut out,
            format!("blocks.{i}.attn.o.weight"),
            &wo_attn,
            4096,
            4096,
        );
        let v_f16: Vec<f16> = matmul_nt_f16_in(&norm1, &wv, 256, 4096, 4096)
            .into_iter()
            .map(|v| v as f16)
            .collect();
        let attn_delta = matmul_nt_f16_in(&v_f16, &wo_attn, 256, 4096, 4096);
        for idx in 0..x.len() {
            x[idx] = ((x[idx] as f32) + attn_delta[idx]) as f16;
        }
        let (_, mut ln2_g) = raw.read_f32(&format!("{p}.layer.1.layer_norm.weight"));
        let (_, mut wi_0) = raw.read_f32(&format!("{p}.layer.1.DenseReluDense.wi_0.weight"));
        let (_, mut wi_1) = raw.read_f32(&format!("{p}.layer.1.DenseReluDense.wi_1.weight"));
        let (_, mut wo_ffn) = raw.read_f32(&format!("{p}.layer.1.DenseReluDense.wo.weight"));
        for v in &mut wo_ffn {
            *v /= res_scale;
        }
        let ln2_test_g: Vec<f16> = ln2_g.iter().map(|&v| v as f16).collect();
        let norm2_test = rmsnorm_fp16(&x, &ln2_test_g, 256, 4096);
        if norm2_test
            .iter()
            .any(|&v| v.is_infinite() || (v as f32).abs() > 8000.0)
        {
            for v in &mut ln2_g {
                *v /= 8.0;
            }
            for v in &mut wi_0 {
                *v *= 8.0;
            }
            for v in &mut wi_1 {
                *v *= 8.0;
            }
        }
        let ln2_g_f16: Vec<f16> = ln2_g
            .iter()
            .map(|&v| v.clamp(-65504.0, 65504.0) as f16)
            .collect();
        let norm2 = rmsnorm_fp16(&x, &ln2_g_f16, 256, 4096);
        let mut u = matmul_nt_f16_in(&norm2, &wi_0, 256, 10240, 4096);
        for v in &mut u {
            let c = -2.3019857 * (*v + 0.044715 * (*v) * (*v) * (*v));
            *v /= 1.0 + c.exp2();
        }
        let g = matmul_nt_f16_in(&norm2, &wi_1, 256, 10240, 4096);
        let ffn_raw_max = u
            .iter()
            .zip(g.iter())
            .fold(0.0f32, |m, (&a, &b)| m.max((a * b).abs()));
        let wo_max = wo_ffn.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let mut ffn_div = 1.0f32;
        while (ffn_raw_max / ffn_div > 8000.0) && (wo_max * ffn_div * 2.0 < 30000.0) {
            ffn_div *= 2.0;
        }
        for v in &mut wi_1 {
            *v /= ffn_div;
        }
        for v in &mut wo_ffn {
            *v *= ffn_div;
        }
        let g_scaled = matmul_nt_f16_in(&norm2, &wi_1, 256, 10240, 4096);
        let ffn_hid_f16: Vec<f16> = u
            .iter()
            .zip(g_scaled.iter())
            .map(|(&a, &b)| (a * b).clamp(-65504.0, 65504.0) as f16)
            .collect();
        let ffn_delta = matmul_nt_f16_in(&ffn_hid_f16, &wo_ffn, 256, 4096, 10240);
        for idx in 0..x.len() {
            x[idx] = ((x[idx] as f32) + ffn_delta[idx]).clamp(-65504.0, 65504.0) as f16;
        }
        insert_f16(
            &mut out,
            format!("blocks.{i}.ln2.gamma"),
            vec![4096],
            &ln2_g,
        );
        insert_packed_nk(
            &mut out,
            format!("blocks.{i}.ffn.b0_wi0.weight"),
            &wi_0,
            4096,
            10240,
        );
        insert_packed_nk(
            &mut out,
            format!("blocks.{i}.ffn.b1_wi1.weight"),
            &wi_1,
            4096,
            10240,
        );
        insert_packed_nk(
            &mut out,
            format!("blocks.{i}.ffn.wo.weight"),
            &wo_ffn,
            10240,
            4096,
        );
    }
    let (fln_shape, fln_vals) = raw.read_f32("encoder.final_layer_norm.weight");
    insert_f16(&mut out, "final_ln.gamma", fln_shape, &fln_vals);
    save_packed(&out, output_path);
}

fn fold_adaln_bias_6(bias: &mut [f32]) {
    for v in &mut bias[2432..4864] {
        *v += 1.0;
    }
    for v in &mut bias[9728..12160] {
        *v += 1.0;
    }
}

fn fold_adaln_bias_2(bias: &mut [f32]) {
    for v in &mut bias[2432..4864] {
        *v += 1.0;
    }
}

fn convert_mmdit(input_path: &Path, output_path: &Path) {
    let raw_bytes = read(input_path).unwrap();
    let raw = RawFile::from_bytes(&raw_bytes);
    let pfx = ["model.diffusion_model.", ""];
    let mut out = PackedMap::new();
    for (dst, src, k, n) in [
        ("t_embedder.mlp.0", "t_embedder.mlp.0", 256, 2432),
        ("t_embedder.mlp.2", "t_embedder.mlp.2", 2432, 2432),
        ("y_embedder.mlp.0", "y_embedder.mlp.0", 2048, 2432),
        ("y_embedder.mlp.2", "y_embedder.mlp.2", 2432, 2432),
    ] {
        let (_, w) = raw.read_f32(&find_key(&raw, &pfx, &format!("{src}.weight")));
        let (_, b) = raw.read_f32(&find_key(&raw, &pfx, &format!("{src}.bias")));
        insert_f16(
            &mut out,
            format!("{dst}.weight"),
            vec![k, n],
            &transpose_2d(&w, n, k),
        );
        insert_f16(&mut out, format!("{dst}.bias"), vec![n], &b);
    }
    let (_, x_proj_w) = raw.read_f32(&find_key(&raw, &pfx, "x_embedder.proj.weight"));
    let (_, ctx_w) = raw.read_f32(&find_key(&raw, &pfx, "context_embedder.weight"));
    let (_, ctx_b) = raw.read_f32(&find_key(&raw, &pfx, "context_embedder.bias"));
    let (pos_shape, pos_embed) = raw.read_f32(&find_key(&raw, &pfx, "pos_embed"));
    let (_, x_proj_b) = raw.read_f32(&find_key(&raw, &pfx, "x_embedder.proj.bias"));
    let grid_size = (pos_shape[1] as f64).sqrt() as usize;
    let top = (grid_size - 64) / 2;
    let mut fused_pos = Vec::with_capacity(4096 * 2432);
    for r in 0..64 {
        for c in 0..64 {
            let src_idx = ((top + r) * grid_size + (top + c)) * 2432;
            for d in 0..2432 {
                fused_pos.push(pos_embed[src_idx + d] + x_proj_b[d]);
            }
        }
    }
    insert_packed_nk(&mut out, "x_embedder.proj.weight", &x_proj_w, 64, 2432);
    insert_packed_nk(&mut out, "context_embedder.weight", &ctx_w, 4096, 2432);
    insert_f16(&mut out, "context_embedder.bias", vec![2432], &ctx_b);
    insert_f16(
        &mut out,
        "x_embedder.fused_pos_bias",
        vec![4096, 2432],
        &fused_pos,
    );
    let pack_lin = |out: &mut PackedMap, dst_prefix: &str, src_prefix: &str, k: usize, n: usize| {
        let (_, w) = raw.read_f32(&find_key(&raw, &pfx, &format!("{src_prefix}.weight")));
        let (_, b) = raw.read_f32(&find_key(&raw, &pfx, &format!("{src_prefix}.bias")));
        insert_packed_nk(out, format!("{dst_prefix}.weight"), &w, k, n);
        insert_f16(out, format!("{dst_prefix}.bias"), vec![n], &b);
    };
    for i in 0..38 {
        let px = format!("joint_blocks.{i}.x_block");
        let pc = format!("joint_blocks.{i}.context_block");
        let is_last = i == 37;
        let (_, wx_raw) = raw.read_f32(&find_key(
            &raw,
            &pfx,
            &format!("{px}.adaLN_modulation.1.weight"),
        ));
        let (_, mut bx_ada) = raw.read_f32(&find_key(
            &raw,
            &pfx,
            &format!("{px}.adaLN_modulation.1.bias"),
        ));
        let wx_ada = transpose_2d(&wx_raw, 14592, 2432);
        fold_adaln_bias_6(&mut bx_ada);
        if !is_last {
            let (_, wc_raw) = raw.read_f32(&find_key(
                &raw,
                &pfx,
                &format!("{pc}.adaLN_modulation.1.weight"),
            ));
            let (_, mut bc_ada) = raw.read_f32(&find_key(
                &raw,
                &pfx,
                &format!("{pc}.adaLN_modulation.1.bias"),
            ));
            let wc_ada = transpose_2d(&wc_raw, 14592, 2432);
            fold_adaln_bias_6(&mut bc_ada);
            let mut w_stack = Vec::with_capacity(2 * 2432 * 14592);
            w_stack.extend_from_slice(&wx_ada);
            w_stack.extend_from_slice(&wc_ada);
            let mut b_stack = Vec::with_capacity(2 * 14592);
            b_stack.extend_from_slice(&bx_ada);
            b_stack.extend_from_slice(&bc_ada);
            insert_f16(
                &mut out,
                format!("blocks.{i}.adaln_14592.weight"),
                vec![2, 2432, 14592],
                &w_stack,
            );
            insert_f16(
                &mut out,
                format!("blocks.{i}.adaln_14592.bias"),
                vec![2, 14592],
                &b_stack,
            );
        } else {
            let (_, wc_raw) = raw.read_f32(&find_key(
                &raw,
                &pfx,
                &format!("{pc}.adaLN_modulation.1.weight"),
            ));
            let (_, mut bc_ada) = raw.read_f32(&find_key(
                &raw,
                &pfx,
                &format!("{pc}.adaLN_modulation.1.bias"),
            ));
            fold_adaln_bias_2(&mut bc_ada);
            insert_f16(
                &mut out,
                format!("blocks.{i}.x_adaln_14592.weight"),
                vec![2432, 14592],
                &wx_ada,
            );
            insert_f16(
                &mut out,
                format!("blocks.{i}.x_adaln_14592.bias"),
                vec![14592],
                &bx_ada,
            );
            insert_f16(
                &mut out,
                format!("blocks.{i}.ctx_adaln_4864.weight"),
                vec![2432, 4864],
                &transpose_2d(&wc_raw, 4864, 2432),
            );
            insert_f16(
                &mut out,
                format!("blocks.{i}.ctx_adaln_4864.bias"),
                vec![4864],
                &bc_ada,
            );
        }
        pack_lin(
            &mut out,
            &format!("blocks.{i}.x_attn.qkv"),
            &format!("{px}.attn.qkv"),
            2432,
            7296,
        );
        pack_lin(
            &mut out,
            &format!("blocks.{i}.ctx_attn.qkv"),
            &format!("{pc}.attn.qkv"),
            2432,
            7296,
        );
        for (dst_q, src_q) in [
            ("qk_norm.gamma_x_q", format!("{px}.attn.ln_q.weight")),
            ("qk_norm.gamma_x_k", format!("{px}.attn.ln_k.weight")),
            ("qk_norm.gamma_ctx_q", format!("{pc}.attn.ln_q.weight")),
            ("qk_norm.gamma_ctx_k", format!("{pc}.attn.ln_k.weight")),
        ] {
            let (s, v) = raw.read_f32(&find_key(&raw, &pfx, &src_q));
            insert_f16(&mut out, format!("blocks.{i}.{dst_q}"), s, &v);
        }
        pack_lin(
            &mut out,
            &format!("blocks.{i}.x_attn.proj"),
            &format!("{px}.attn.proj"),
            2432,
            2432,
        );
        pack_lin(
            &mut out,
            &format!("blocks.{i}.x_mlp.fc1"),
            &format!("{px}.mlp.fc1"),
            2432,
            9728,
        );
        pack_lin(
            &mut out,
            &format!("blocks.{i}.x_mlp.fc2"),
            &format!("{px}.mlp.fc2"),
            9728,
            2432,
        );
        if !is_last {
            pack_lin(
                &mut out,
                &format!("blocks.{i}.ctx_attn.proj"),
                &format!("{pc}.attn.proj"),
                2432,
                2432,
            );
            pack_lin(
                &mut out,
                &format!("blocks.{i}.ctx_mlp.fc1"),
                &format!("{pc}.mlp.fc1"),
                2432,
                9728,
            );
            pack_lin(
                &mut out,
                &format!("blocks.{i}.ctx_mlp.fc2"),
                &format!("{pc}.mlp.fc2"),
                9728,
                2432,
            );
        }
    }
    let (_, fl_ada_w) = raw.read_f32(&find_key(
        &raw,
        &pfx,
        "final_layer.adaLN_modulation.1.weight",
    ));
    let (_, mut fl_ada_b) =
        raw.read_f32(&find_key(&raw, &pfx, "final_layer.adaLN_modulation.1.bias"));
    let (_, fl_w_raw) = raw.read_f32(&find_key(&raw, &pfx, "final_layer.linear.weight"));
    let (_, fl_b_raw) = raw.read_f32(&find_key(&raw, &pfx, "final_layer.linear.bias"));
    fold_adaln_bias_2(&mut fl_ada_b);
    let mut fl_w_perm = vec![0.0f32; 64 * 2432];
    let mut fl_b_perm = vec![0.0f32; 64];
    for p0 in 0..2 {
        for p1 in 0..2 {
            for c in 0..16 {
                let src_row = (p0 * 2 + p1) * 16 + c;
                let dst_row = (c * 2 + p0) * 2 + p1;
                fl_b_perm[dst_row] = fl_b_raw[src_row];
                fl_w_perm[dst_row * 2432..(dst_row + 1) * 2432]
                    .copy_from_slice(&fl_w_raw[src_row * 2432..(src_row + 1) * 2432]);
            }
        }
    }
    insert_f16(
        &mut out,
        "final_layer.adaln_4864.weight",
        vec![2432, 4864],
        &transpose_2d(&fl_ada_w, 4864, 2432),
    );
    insert_f16(
        &mut out,
        "final_layer.adaln_4864.bias",
        vec![4864],
        &fl_ada_b,
    );
    insert_packed_nk(&mut out, "final_layer.linear.weight", &fl_w_perm, 2432, 64);
    insert_f16(&mut out, "final_layer.linear.bias", vec![64], &fl_b_perm);
    save_packed(&out, output_path);
}

fn convert_qwen3(paths: &[PathBuf], output_path: &Path) {
    let buffers: Vec<Vec<u8>> = paths.iter().map(|p| read(p).unwrap()).collect();
    let raw = RawSet::from_buffers(&buffers);
    let pfx = ["model.", ""];
    let mut out = PackedMap::new();
    let res_scale = 8.0f32;
    let (emb_shape, mut emb_vals) = raw.read_f32(&find_key_set(&raw, &pfx, "embed_tokens.weight"));
    for v in &mut emb_vals {
        *v = (*v / res_scale) as f16 as f32;
    }
    insert_f16(&mut out, "embed_tokens.weight", emb_shape, &emb_vals);
    let calib_prefix = [
        151644usize,
        872,
        198,
        8251,
        11,
        264,
        10765,
        315,
        264,
        11202,
        3691,
        304,
        6633,
        1189,
        11,
        6399,
        10300,
        11,
        825,
        323,
        825,
        151645,
        198,
        151644,
        77091,
        198,
    ];
    let mut x = vec![0.0f16; 256 * 2560];
    for r in 0..256 {
        let tok = *calib_prefix.get(r).unwrap_or(&151643);
        let src = &emb_vals[tok * 2560..(tok + 1) * 2560];
        for c in 0..2560 {
            x[r * 2560 + c] = src[c] as f16;
        }
    }
    let sqrt_k = (2560.0f32).sqrt();
    for i in 0..35 {
        let p = format!("layers.{i}");
        let (_, mut ln1) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{p}.input_layernorm.weight"),
        ));
        let (_, q_norm) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{p}.self_attn.q_norm.weight"),
        ));
        let (_, k_norm) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{p}.self_attn.k_norm.weight"),
        ));
        let (_, mut wq) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{p}.self_attn.q_proj.weight"),
        ));
        let (_, mut wk) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{p}.self_attn.k_proj.weight"),
        ));
        let (_, mut wv) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{p}.self_attn.v_proj.weight"),
        ));
        let (_, mut wo) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{p}.self_attn.o_proj.weight"),
        ));
        for v in &mut wo {
            *v /= res_scale;
        }
        loop {
            let ln1_test: Vec<f16> = ln1
                .iter()
                .map(|&v| v.clamp(-65504.0, 65504.0) as f16)
                .collect();
            let norm1_test = rmsnorm_fp16(&x, &ln1_test, 256, 2560);
            if !norm1_test
                .iter()
                .any(|&v| v.is_infinite() || (v as f32).abs() > 2000.0)
            {
                break;
            }
            for v in &mut ln1 {
                *v /= 8.0;
            }
            for v in &mut wq {
                *v *= 8.0;
            }
            for v in &mut wk {
                *v *= 8.0;
            }
            for v in &mut wv {
                *v *= 8.0;
            }
        }
        let ln1_f16: Vec<f16> = ln1
            .iter()
            .map(|&v| v.clamp(-65504.0, 65504.0) as f16)
            .collect();
        let norm1 = rmsnorm_fp16(&x, &ln1_f16, 256, 2560);
        let ln1_rms = (ln1.iter().map(|&v| v * v).sum::<f32>() / 2560.0).sqrt();
        let q_sim = matmul_nt_f16_in(&norm1, &wq, 256, 4096, 2560);
        let q_max = q_sim.iter().fold(
            max_row_l2(&wq, 4096, 2560) * sqrt_k * ln1_rms * 0.5,
            |m, &v| m.max(v.abs()),
        );
        let mut q_div = 1.0f32;
        while q_max / q_div > 2000.0 {
            q_div *= 2.0;
        }
        if q_div > 1.0 {
            for v in &mut wq {
                *v /= q_div;
            }
        }
        let k_sim = matmul_nt_f16_in(&norm1, &wk, 256, 1024, 2560);
        let k_max = k_sim.iter().fold(
            max_row_l2(&wk, 1024, 2560) * sqrt_k * ln1_rms * 0.5,
            |m, &v| m.max(v.abs()),
        );
        let mut k_div = 1.0f32;
        while k_max / k_div > 2000.0 {
            k_div *= 2.0;
        }
        if k_div > 1.0 {
            for v in &mut wk {
                *v /= k_div;
            }
        }
        let v_sim = matmul_nt_f16_in(&norm1, &wv, 256, 1024, 2560);
        let v_max = v_sim.iter().fold(
            max_row_l2(&wv, 1024, 2560) * sqrt_k * ln1_rms * 0.5,
            |m, &v| m.max(v.abs()),
        );
        let wo_max = wo.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let mut v_div = 1.0f32;
        while (v_max / v_div > 2000.0) && (wo_max * v_div * 2.0 < 30000.0) {
            v_div *= 2.0;
        }
        if v_div > 1.0 {
            for v in &mut wv {
                *v /= v_div;
            }
            for v in &mut wo {
                *v *= v_div;
            }
        }
        let q_scaled = matmul_nt_f16_in(&norm1, &wq, 256, 4096, 2560);
        let k_scaled = matmul_nt_f16_in(&norm1, &wk, 256, 1024, 2560);
        let v_scaled = matmul_nt_f16_in(&norm1, &wv, 256, 1024, 2560);
        let attn_in = qwen3_causal_gqa_f32(&q_scaled, &k_scaled, &v_scaled, &q_norm, &k_norm);
        let attn_delta = matmul_nt_f16_in(&attn_in, &wo, 256, 2560, 4096);
        for idx in 0..x.len() {
            x[idx] = ((x[idx] as f32) + attn_delta[idx]).clamp(-65504.0, 65504.0) as f16;
        }
        let (_, mut ln2) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{p}.post_attention_layernorm.weight"),
        ));
        let (_, mut w_gate) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{p}.mlp.gate_proj.weight"),
        ));
        let (_, mut w_up) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{p}.mlp.up_proj.weight"),
        ));
        let (_, mut w_down) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{p}.mlp.down_proj.weight"),
        ));
        for v in &mut w_down {
            *v /= res_scale;
        }
        loop {
            let ln2_test: Vec<f16> = ln2
                .iter()
                .map(|&v| v.clamp(-65504.0, 65504.0) as f16)
                .collect();
            let norm2_test = rmsnorm_fp16(&x, &ln2_test, 256, 2560);
            if !norm2_test
                .iter()
                .any(|&v| v.is_infinite() || (v as f32).abs() > 2000.0)
            {
                break;
            }
            for v in &mut ln2 {
                *v /= 8.0;
            }
            for v in &mut w_gate {
                *v *= 8.0;
            }
            for v in &mut w_up {
                *v *= 8.0;
            }
        }
        let ln2_f16: Vec<f16> = ln2
            .iter()
            .map(|&v| v.clamp(-65504.0, 65504.0) as f16)
            .collect();
        let norm2 = rmsnorm_fp16(&x, &ln2_f16, 256, 2560);
        let ln2_rms = (ln2.iter().map(|&v| v * v).sum::<f32>() / 2560.0).sqrt();
        let mut gate_act = matmul_nt_f16_in(&norm2, &w_gate, 256, 9728, 2560);
        for v in &mut gate_act {
            *v /= 1.0 + (-*v).exp();
        }
        let up_act = matmul_nt_f16_in(&norm2, &w_up, 256, 9728, 2560);
        let gate_bound = max_row_l2(&w_gate, 9728, 2560) * sqrt_k * ln2_rms * 0.5;
        let up_bound = max_row_l2(&w_up, 9728, 2560) * sqrt_k * ln2_rms * 0.5;
        let ffn_raw_max = gate_act
            .iter()
            .zip(up_act.iter())
            .fold(up_bound.max(gate_bound * up_bound * 0.25), |m, (&a, &b)| {
                m.max((a * b).abs()).max(b.abs())
            });
        let down_max = w_down.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let mut ffn_div = 1.0f32;
        while (ffn_raw_max / ffn_div > 2000.0) && (down_max * ffn_div * 2.0 < 30000.0) {
            ffn_div *= 2.0;
        }
        if ffn_div > 1.0 {
            for v in &mut w_up {
                *v /= ffn_div;
            }
            for v in &mut w_down {
                *v *= ffn_div;
            }
        }
        let ffn_hid_f16: Vec<f16> = gate_act
            .iter()
            .zip(up_act.iter())
            .map(|(&a, &b)| ((a * b) / ffn_div).clamp(-65504.0, 65504.0) as f16)
            .collect();
        let ffn_delta = matmul_nt_f16_in(&ffn_hid_f16, &w_down, 256, 2560, 9728);
        for idx in 0..x.len() {
            x[idx] = ((x[idx] as f32) + ffn_delta[idx]).clamp(-65504.0, 65504.0) as f16;
        }
        insert_f16(&mut out, format!("layers.{i}.ln1.gamma"), vec![2560], &ln1);
        insert_f16(
            &mut out,
            format!("layers.{i}.attn.q_norm"),
            vec![128],
            &q_norm,
        );
        insert_f16(
            &mut out,
            format!("layers.{i}.attn.k_norm"),
            vec![128],
            &k_norm,
        );
        let mut wqkv = Vec::with_capacity(6144 * 2560);
        wqkv.extend_from_slice(&wq);
        wqkv.extend_from_slice(&wk);
        wqkv.extend_from_slice(&wv);
        insert_packed_nk(
            &mut out,
            format!("layers.{i}.attn.qkv.weight"),
            &wqkv,
            2560,
            6144,
        );
        insert_packed_nk(
            &mut out,
            format!("layers.{i}.attn.o.weight"),
            &wo,
            4096,
            2560,
        );
        insert_f16(&mut out, format!("layers.{i}.ln2.gamma"), vec![2560], &ln2);
        insert_packed_nk(
            &mut out,
            format!("layers.{i}.mlp.gate.weight"),
            &w_gate,
            2560,
            9728,
        );
        insert_packed_nk(
            &mut out,
            format!("layers.{i}.mlp.up.weight"),
            &w_up,
            2560,
            9728,
        );
        insert_packed_nk(
            &mut out,
            format!("layers.{i}.mlp.down.weight"),
            &w_down,
            9728,
            2560,
        );
    }
    save_packed(&out, output_path);
}

fn convert_s3dit(paths: &[PathBuf], output_path: &Path) {
    let buffers: Vec<Vec<u8>> = paths.iter().map(|p| read(p).unwrap()).collect();
    let raw = RawSet::from_buffers(&buffers);
    let pfx = ["", "model.diffusion_model."];
    let mut out = PackedMap::new();
    for (name, k, n) in [
        ("t_embedder.mlp.0", 256, 1024),
        ("t_embedder.mlp.2", 1024, 256),
    ] {
        let (_, w) = raw.read_f32(&find_key_set(&raw, &pfx, &format!("{name}.weight")));
        let (_, b) = raw.read_f32(&find_key_set(&raw, &pfx, &format!("{name}.bias")));
        insert_f16(
            &mut out,
            format!("{name}.weight"),
            vec![k, n],
            &transpose_2d(&w, n, k),
        );
        insert_f16(&mut out, format!("{name}.bias"), vec![n], &b);
    }
    let (_, cap_norm) = raw.read_f32(&find_key_set(&raw, &pfx, "cap_embedder.0.weight"));
    let (_, cap_lin_w) = raw.read_f32(&find_key_set(&raw, &pfx, "cap_embedder.1.weight"));
    let (_, cap_lin_b) = raw.read_f32(&find_key_set(&raw, &pfx, "cap_embedder.1.bias"));
    let (_, cap_pad) = raw.read_f32(&find_key_set(&raw, &pfx, "cap_pad_token"));
    insert_f16(&mut out, "cap_embedder.norm.gamma", vec![2560], &cap_norm);
    insert_packed_nk(
        &mut out,
        "cap_embedder.linear.weight",
        &cap_lin_w,
        2560,
        3840,
    );
    insert_f16(&mut out, "cap_embedder.linear.bias", vec![3840], &cap_lin_b);
    insert_f16(&mut out, "cap_pad_token", vec![3840], &cap_pad);
    let (_, x_w_raw) = raw.read_f32(&find_key_set(&raw, &pfx, "all_x_embedder.2-1.weight"));
    let (_, x_b) = raw.read_f32(&find_key_set(&raw, &pfx, "all_x_embedder.2-1.bias"));
    let mut x_w_perm = vec![0.0f32; 3840 * 64];
    for row in 0..3840 {
        for p0 in 0..2 {
            for p1 in 0..2 {
                for c in 0..16 {
                    let src_col = (p0 * 2 + p1) * 16 + c;
                    let dst_col = (c * 2 + p0) * 2 + p1;
                    x_w_perm[row * 64 + dst_col] = x_w_raw[row * 64 + src_col];
                }
            }
        }
    }
    insert_packed_nk(&mut out, "x_embedder.weight", &x_w_perm, 64, 3840);
    insert_f16(&mut out, "x_embedder.bias", vec![3840], &x_b);
    let mut pack_block = |src_p: &str, dst_p: &str, modulated: bool| {
        let (_, an1) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{src_p}.attention_norm1.weight"),
        ));
        let (_, an2) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{src_p}.attention_norm2.weight"),
        ));
        let (_, fn1) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{src_p}.ffn_norm1.weight"),
        ));
        let (_, fn2) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{src_p}.ffn_norm2.weight"),
        ));
        insert_f16(&mut out, format!("{dst_p}.attn_norm1"), vec![3840], &an1);
        insert_f16(&mut out, format!("{dst_p}.attn_norm2"), vec![3840], &an2);
        insert_f16(&mut out, format!("{dst_p}.ffn_norm1"), vec![3840], &fn1);
        insert_f16(&mut out, format!("{dst_p}.ffn_norm2"), vec![3840], &fn2);
        let (_, mut wq) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{src_p}.attention.to_q.weight"),
        ));
        let (_, mut wk) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{src_p}.attention.to_k.weight"),
        ));
        let (_, mut wv) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{src_p}.attention.to_v.weight"),
        ));
        let (_, nq) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{src_p}.attention.norm_q.weight"),
        ));
        let (_, nk) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{src_p}.attention.norm_k.weight"),
        ));
        let (_, mut wo) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{src_p}.attention.to_out.0.weight"),
        ));
        let (_, w1) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{src_p}.feed_forward.w1.weight"),
        ));
        let (_, mut w3) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{src_p}.feed_forward.w3.weight"),
        ));
        let (_, mut w2) = raw.read_f32(&find_key_set(
            &raw,
            &pfx,
            &format!("{src_p}.feed_forward.w2.weight"),
        ));
        let (msa_scale_est, mlp_scale_est) = if modulated {
            let (_, ada_w) = raw.read_f32(&find_key_set(
                &raw,
                &pfx,
                &format!("{src_p}.adaLN_modulation.0.weight"),
            ));
            let (_, ada_b) = raw.read_f32(&find_key_set(
                &raw,
                &pfx,
                &format!("{src_p}.adaLN_modulation.0.bias"),
            ));
            let mut msa_sq = 0.0f64;
            for c in 0..3840 {
                let row = &ada_w[c * 256..(c + 1) * 256];
                let l2 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
                let s = (1.0 + ada_b[c]).abs() + l2;
                msa_sq += (s as f64) * (s as f64);
            }
            let mut mlp_sq = 0.0f64;
            for c in 7680..11520 {
                let row = &ada_w[c * 256..(c + 1) * 256];
                let l2 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
                let s = (1.0 + ada_b[c]).abs() + l2;
                mlp_sq += (s as f64) * (s as f64);
            }
            insert_f16(
                &mut out,
                format!("{dst_p}.adaln.weight"),
                vec![256, 15360],
                &transpose_2d(&ada_w, 15360, 256),
            );
            insert_f16(&mut out, format!("{dst_p}.adaln.bias"), vec![15360], &ada_b);
            (
                ((msa_sq / 3840.0).sqrt() as f32).max(1.0),
                ((mlp_sq / 3840.0).sqrt() as f32).max(1.0),
            )
        } else {
            (1.0f32, 1.0f32)
        };
        let an1_rms = (an1.iter().map(|&v| v * v).sum::<f32>() / 3840.0)
            .sqrt()
            .max(0.25)
            * msa_scale_est;
        let q_div = pow2_safe_div(
            rms_gain(&wq, 3840) * an1_rms,
            max_row_l2(&wq, 3840, 3840) * an1_rms,
            2.0,
            32.0,
        );
        let k_div = pow2_safe_div(
            rms_gain(&wk, 3840) * an1_rms,
            max_row_l2(&wk, 3840, 3840) * an1_rms,
            2.0,
            32.0,
        );
        let v_div = pow2_safe_div(
            rms_gain(&wv, 3840) * an1_rms,
            max_row_l2(&wv, 3840, 3840) * an1_rms,
            2.0,
            32.0,
        );
        let v_out_rms = (rms_gain(&wv, 3840) * an1_rms / v_div).max(0.5);
        let o_div = pow2_safe_div(
            rms_gain(&wo, 3840) * v_out_rms,
            max_row_l2(&wo, 3840, 3840) * v_out_rms,
            2.0,
            32.0,
        );
        let fn1_rms = (fn1.iter().map(|&v| v * v).sum::<f32>() / 3840.0)
            .sqrt()
            .max(0.25)
            * mlp_scale_est;
        let w1_gain = (rms_gain(&w1, 10240) * fn1_rms).max(1.0);
        let w1_peak = (max_row_l2(&w1, 10240, 3840) * fn1_rms).max(1.0);
        let w3_div = pow2_safe_div(
            rms_gain(&w3, 10240) * fn1_rms * w1_gain,
            max_row_l2(&w3, 10240, 3840) * fn1_rms * w1_peak,
            2.0,
            64.0,
        );
        let mlp_hid_rms = (rms_gain(&w3, 10240) * fn1_rms * w1_gain / w3_div).max(0.5);
        let w2_div = pow2_safe_div(
            rms_gain(&w2, 3840) * mlp_hid_rms,
            max_row_l2(&w2, 3840, 10240) * mlp_hid_rms,
            2.0,
            32.0,
        );
        if q_div > 1.0 {
            for v in &mut wq {
                *v /= q_div;
            }
        }
        if k_div > 1.0 {
            for v in &mut wk {
                *v /= k_div;
            }
        }
        if v_div > 1.0 {
            for v in &mut wv {
                *v /= v_div;
            }
        }
        if o_div > 1.0 {
            for v in &mut wo {
                *v /= o_div;
            }
        }
        if w3_div > 1.0 {
            for v in &mut w3 {
                *v /= w3_div;
            }
        }
        if w2_div > 1.0 {
            for v in &mut w2 {
                *v /= w2_div;
            }
        }
        let mut wqkv = Vec::with_capacity(11520 * 3840);
        wqkv.extend_from_slice(&wq);
        wqkv.extend_from_slice(&wk);
        wqkv.extend_from_slice(&wv);
        insert_packed_nk(
            &mut out,
            format!("{dst_p}.attn.qkv.weight"),
            &wqkv,
            3840,
            11520,
        );
        insert_f16(&mut out, format!("{dst_p}.attn.norm_q"), vec![128], &nq);
        insert_f16(&mut out, format!("{dst_p}.attn.norm_k"), vec![128], &nk);
        insert_packed_nk(
            &mut out,
            format!("{dst_p}.attn.out.weight"),
            &wo,
            3840,
            3840,
        );
        insert_packed_nk(&mut out, format!("{dst_p}.ffn.w1.weight"), &w1, 3840, 10240);
        insert_packed_nk(&mut out, format!("{dst_p}.ffn.w3.weight"), &w3, 3840, 10240);
        insert_packed_nk(&mut out, format!("{dst_p}.ffn.w2.weight"), &w2, 10240, 3840);
    };
    for i in 0..2 {
        pack_block(
            &format!("context_refiner.{i}"),
            &format!("context_refiner.{i}"),
            false,
        );
        pack_block(
            &format!("noise_refiner.{i}"),
            &format!("noise_refiner.{i}"),
            true,
        );
    }
    for i in 0..30 {
        pack_block(&format!("layers.{i}"), &format!("layers.{i}"), true);
    }
    let (_, fl_ada_w) = raw.read_f32(&find_key_set(
        &raw,
        &pfx,
        "all_final_layer.2-1.adaLN_modulation.1.weight",
    ));
    let (_, fl_ada_b) = raw.read_f32(&find_key_set(
        &raw,
        &pfx,
        "all_final_layer.2-1.adaLN_modulation.1.bias",
    ));
    let (_, fl_w_raw) = raw.read_f32(&find_key_set(
        &raw,
        &pfx,
        "all_final_layer.2-1.linear.weight",
    ));
    let (_, fl_b_raw) = raw.read_f32(&find_key_set(&raw, &pfx, "all_final_layer.2-1.linear.bias"));
    let mut fl_w_perm = vec![0.0f32; 64 * 3840];
    let mut fl_b_perm = vec![0.0f32; 64];
    for p0 in 0..2 {
        for p1 in 0..2 {
            for c in 0..16 {
                let src_row = (p0 * 2 + p1) * 16 + c;
                let dst_row = (c * 2 + p0) * 2 + p1;
                fl_b_perm[dst_row] = fl_b_raw[src_row];
                fl_w_perm[dst_row * 3840..(dst_row + 1) * 3840]
                    .copy_from_slice(&fl_w_raw[src_row * 3840..(src_row + 1) * 3840]);
            }
        }
    }
    insert_f16(
        &mut out,
        "final_layer.adaln.weight",
        vec![256, 3840],
        &transpose_2d(&fl_ada_w, 3840, 256),
    );
    insert_f16(&mut out, "final_layer.adaln.bias", vec![3840], &fl_ada_b);
    insert_packed_nk(&mut out, "final_layer.linear.weight", &fl_w_perm, 3840, 64);
    insert_f16(&mut out, "final_layer.linear.bias", vec![64], &fl_b_perm);
    save_packed(&out, output_path);
}

fn convert_vae(input_path: &Path, output_path: &Path) {
    let raw_bytes = read(input_path).unwrap();
    let raw = RawFile::from_bytes(&raw_bytes);
    let pfx = ["first_stage_model.", ""];
    let is_diffusers = pfx
        .iter()
        .any(|p| raw.has(&format!("{p}decoder.up_blocks.0.resnets.0.conv1.weight")));
    let mut out = PackedMap::new();
    let get_wb = |diff_key: &str, ldm_key: &str| -> (Vec<usize>, Vec<f32>, Vec<f32>) {
        let base = if is_diffusers { diff_key } else { ldm_key };
        let (ws, w) = raw.read_f32(&find_key(&raw, &pfx, &format!("{base}.weight")));
        let (_, b) = raw.read_f32(&find_key(&raw, &pfx, &format!("{base}.bias")));
        (ws, w, b)
    };
    let add_conv3x3 = |out: &mut PackedMap, name: &str, diff_key: &str, ldm_key: &str| {
        let (ws, w, b) = get_wb(diff_key, ldm_key);
        let (f_out, c_in) = (ws[0], ws[1]);
        let mut w_perm = vec![0.0f32; 9 * c_in * f_out];
        for f in 0..f_out {
            for c in 0..c_in {
                for kh in 0..3 {
                    for kw in 0..3 {
                        let src = ((f * c_in + c) * 3 + kh) * 3 + kw;
                        let dst = (((kh * 3 + kw) * c_in + c) * f_out) + f;
                        w_perm[dst] = w[src];
                    }
                }
            }
        }
        out.insert(
            format!("{name}.weight"),
            (
                vec![9 * c_in, f_out],
                pack_b_tiles_16x16(&w_perm, 9 * c_in, f_out),
            ),
        );
        insert_f16(out, format!("{name}.bias"), vec![f_out], &b);
    };
    let add_conv1x1 = |out: &mut PackedMap, name: &str, diff_key: &str, ldm_key: &str| {
        let (ws, w, b) = get_wb(diff_key, ldm_key);
        let (f_out, c_in) = (ws[0], ws[1]);
        insert_packed_nk(out, format!("{name}.weight"), &w, c_in, f_out);
        insert_f16(out, format!("{name}.bias"), vec![f_out], &b);
    };
    let add_gn = |out: &mut PackedMap, name: &str, diff_key: &str, ldm_key: &str| {
        let (ws, w, b) = get_wb(diff_key, ldm_key);
        insert_f16(out, format!("{name}.gamma"), ws.clone(), &w);
        insert_f16(out, format!("{name}.beta"), ws, &b);
    };
    add_conv3x3(&mut out, "conv_in", "decoder.conv_in", "decoder.conv_in");
    add_gn(
        &mut out,
        "mid.block_1.norm1",
        "decoder.mid_block.resnets.0.norm1",
        "decoder.mid.block_1.norm1",
    );
    add_conv3x3(
        &mut out,
        "mid.block_1.conv1",
        "decoder.mid_block.resnets.0.conv1",
        "decoder.mid.block_1.conv1",
    );
    add_gn(
        &mut out,
        "mid.block_1.norm2",
        "decoder.mid_block.resnets.0.norm2",
        "decoder.mid.block_1.norm2",
    );
    add_conv3x3(
        &mut out,
        "mid.block_1.conv2",
        "decoder.mid_block.resnets.0.conv2",
        "decoder.mid.block_1.conv2",
    );
    add_gn(
        &mut out,
        "mid.attn.norm",
        "decoder.mid_block.attentions.0.group_norm",
        "decoder.mid.attn_1.norm",
    );
    let (_, wq, bq) = get_wb(
        "decoder.mid_block.attentions.0.to_q",
        "decoder.mid.attn_1.q",
    );
    let (_, wk, bk) = get_wb(
        "decoder.mid_block.attentions.0.to_k",
        "decoder.mid.attn_1.k",
    );
    let (_, wv, bv) = get_wb(
        "decoder.mid_block.attentions.0.to_v",
        "decoder.mid.attn_1.v",
    );
    let mut qkv_w = Vec::with_capacity(1536 * 512);
    qkv_w.extend_from_slice(&wq);
    qkv_w.extend_from_slice(&wk);
    qkv_w.extend_from_slice(&wv);
    let mut qkv_b = Vec::with_capacity(1536);
    qkv_b.extend_from_slice(&bq);
    qkv_b.extend_from_slice(&bk);
    qkv_b.extend_from_slice(&bv);
    insert_packed_nk(&mut out, "mid.attn.qkv.weight", &qkv_w, 512, 1536);
    insert_f16(&mut out, "mid.attn.qkv.bias", vec![1536], &qkv_b);
    add_conv1x1(
        &mut out,
        "mid.attn.proj_out",
        "decoder.mid_block.attentions.0.to_out.0",
        "decoder.mid.attn_1.proj_out",
    );
    add_gn(
        &mut out,
        "mid.block_2.norm1",
        "decoder.mid_block.resnets.1.norm1",
        "decoder.mid.block_2.norm1",
    );
    add_conv3x3(
        &mut out,
        "mid.block_2.conv1",
        "decoder.mid_block.resnets.1.conv1",
        "decoder.mid.block_2.conv1",
    );
    add_gn(
        &mut out,
        "mid.block_2.norm2",
        "decoder.mid_block.resnets.1.norm2",
        "decoder.mid.block_2.norm2",
    );
    add_conv3x3(
        &mut out,
        "mid.block_2.conv2",
        "decoder.mid_block.resnets.1.conv2",
        "decoder.mid.block_2.conv2",
    );
    for up_idx in 0..4 {
        let ldm_up_idx = 3 - up_idx;
        for res_idx in 0..3 {
            let p_diff = format!("decoder.up_blocks.{up_idx}.resnets.{res_idx}");
            let p_ldm = format!("decoder.up.{ldm_up_idx}.block.{res_idx}");
            let prefix = format!("up.{up_idx}.res.{res_idx}");
            add_gn(
                &mut out,
                &format!("{prefix}.norm1"),
                &format!("{p_diff}.norm1"),
                &format!("{p_ldm}.norm1"),
            );
            add_conv3x3(
                &mut out,
                &format!("{prefix}.conv1"),
                &format!("{p_diff}.conv1"),
                &format!("{p_ldm}.conv1"),
            );
            add_gn(
                &mut out,
                &format!("{prefix}.norm2"),
                &format!("{p_diff}.norm2"),
                &format!("{p_ldm}.norm2"),
            );
            add_conv3x3(
                &mut out,
                &format!("{prefix}.conv2"),
                &format!("{p_diff}.conv2"),
                &format!("{p_ldm}.conv2"),
            );
            if (up_idx == 2 || up_idx == 3) && res_idx == 0 {
                add_conv1x1(
                    &mut out,
                    &format!("{prefix}.conv_shortcut"),
                    &format!("{p_diff}.conv_shortcut"),
                    &format!("{p_ldm}.nin_shortcut"),
                );
            }
        }
        if up_idx < 3 {
            add_conv3x3(
                &mut out,
                &format!("up.{up_idx}.upsample_conv"),
                &format!("decoder.up_blocks.{up_idx}.upsamplers.0.conv"),
                &format!("decoder.up.{ldm_up_idx}.upsample.conv"),
            );
        }
    }
    add_gn(
        &mut out,
        "norm_out",
        "decoder.conv_norm_out",
        "decoder.norm_out",
    );
    let (_, w_out, mut b_out) = get_wb("decoder.conv_out", "decoder.conv_out");
    let mut w_out_perm = vec![0.0f32; 3 * 3 * 3 * 128];
    for f in 0..3 {
        for c in 0..128 {
            for kh in 0..3 {
                for kw in 0..3 {
                    let src = ((f * 128 + c) * 3 + kh) * 3 + kw;
                    let dst = (((f * 3 + kh) * 3 + kw) * 128) + c;
                    w_out_perm[dst] = w_out[src] * 127.5;
                }
            }
        }
    }
    for v in &mut b_out {
        *v = *v * 127.5 + 127.5;
    }
    insert_f16(&mut out, "conv_out.weight", vec![3, 3, 3, 128], &w_out_perm);
    insert_f16(&mut out, "conv_out.bias", vec![3], &b_out);
    save_packed(&out, output_path);
}

fn find_local(filename: &str, raw_dirs: &[PathBuf]) -> Option<PathBuf> {
    let basename = Path::new(filename).file_name().unwrap();
    for d in raw_dirs {
        for cand in [d.join(filename), d.join(basename)] {
            if cand.exists() {
                return Some(cand);
            }
        }
    }
    None
}

fn fetch(
    client: &HFClientSync,
    owner: &str,
    name: &str,
    filename: &str,
    raw_dirs: &[PathBuf],
) -> PathBuf {
    if let Some(local) = find_local(filename, raw_dirs) {
        return local;
    }
    print!("downloading {filename}... ");
    stdout().flush().unwrap();
    client
        .model(owner, name)
        .download_file()
        .filename(filename)
        .send()
        .unwrap()
}

fn extract_shards_from_index(index_path: &Path) -> Vec<String> {
    let text = read_to_string(index_path).unwrap();
    let wm_start = text.find("\"weight_map\"").unwrap_or(0);
    let mut shards = BTreeSet::new();
    for seg in text[wm_start..].split('"') {
        if seg.ends_with(".safetensors") {
            let name = Path::new(seg).file_name().unwrap().to_str().unwrap();
            shards.insert(name.to_string());
        }
    }
    assert!(
        !shards.is_empty(),
        "no safetensors shards found in {}",
        index_path.display()
    );
    shards.into_iter().collect()
}

fn fetch_sharded(
    client: &HFClientSync,
    owner: &str,
    name: &str,
    subdir: &str,
    single_file: &str,
    index_file: &str,
    raw_dirs: &[PathBuf],
) -> Vec<PathBuf> {
    let single_rel = format!("{subdir}/{single_file}");
    if let Some(p) = find_local(&single_rel, raw_dirs) {
        return vec![p];
    }
    let idx_rel = format!("{subdir}/{index_file}");
    let idx_path = fetch(client, owner, name, &idx_rel, raw_dirs);
    extract_shards_from_index(&idx_path)
        .into_iter()
        .map(|s| fetch(client, owner, name, &format!("{subdir}/{s}"), raw_dirs))
        .collect()
}

fn run_step(idx: usize, total: usize, dst: &Path, force: bool, build: impl FnOnce()) {
    let name = dst.file_name().unwrap().to_str().unwrap();
    print!("[{idx}/{total}] {name:<22} ");
    stdout().flush().unwrap();
    if !force && dst.exists() {
        println!("skipped");
        return;
    }
    let t0 = Instant::now();
    build();
    println!("done ({:.2}s)", t0.elapsed().as_secs_f32());
}

pub fn run(args: &[String]) {
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help") {
        println!(
            "Usage: {} install <sd35|zimage> [--force] [--raw-dir <DIR>]",
            env!("CARGO_PKG_NAME")
        );
        if args.is_empty() {
            exit(1);
        }
        return;
    }
    let model = match args[0].as_str() {
        "sd35" => "sd35",
        "zimage" => "zimage",
        other => {
            eprintln!("Error: unknown model `{other}` (expected `sd35` or `zimage`)");
            exit(1);
        }
    };
    let mut force = false;
    let mut search_dirs = Vec::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: {} install <sd35|zimage> [--force] [--raw-dir <DIR>]",
                    env!("CARGO_PKG_NAME")
                );
                return;
            }
            "--force" => force = true,
            "--raw-dir" => {
                i += 1;
                let Some(dir) = args.get(i) else {
                    eprintln!("Error: missing value for --raw-dir");
                    exit(1);
                };
                search_dirs.push(PathBuf::from(dir));
            }
            other => {
                eprintln!("Error: unknown flag `{other}`");
                exit(1);
            }
        }
        i += 1;
    }
    let out_dir = crate::engine::get_data_dir().join(model);
    create_dir_all(&out_dir).unwrap();
    let client = HFClientSync::new().unwrap();
    let t_all = Instant::now();
    if model == "sd35" {
        search_dirs.push(PathBuf::from("sd35_turbo_raw"));
        let clip_l_out = out_dir.join("clip_l.safetensors");
        let clip_g_out = out_dir.join("clip_g.safetensors");
        let t5_out = out_dir.join("t5xxl.safetensors");
        let mmdit_out = out_dir.join("mmdit.safetensors");
        let vae_out = out_dir.join("vae.safetensors");
        let mut cached_ckpt: Option<PathBuf> = None;
        run_step(1, 5, &clip_l_out, force, || {
            let src = fetch(
                &client,
                SD35_OWNER,
                SD35_NAME,
                "text_encoders/clip_l.safetensors",
                &search_dirs,
            );
            convert_clip(&src, &clip_l_out, 768, 12);
        });
        run_step(2, 5, &clip_g_out, force, || {
            let src = fetch(
                &client,
                SD35_OWNER,
                SD35_NAME,
                "text_encoders/clip_g.safetensors",
                &search_dirs,
            );
            convert_clip(&src, &clip_g_out, 1280, 32);
        });
        run_step(3, 5, &t5_out, force, || {
            let src = fetch(
                &client,
                SD35_OWNER,
                SD35_NAME,
                "text_encoders/t5xxl_fp16.safetensors",
                &search_dirs,
            );
            convert_t5(&src, &t5_out);
        });
        run_step(4, 5, &mmdit_out, force, || {
            let src = fetch(
                &client,
                SD35_OWNER,
                SD35_NAME,
                "sd3.5_large_turbo.safetensors",
                &search_dirs,
            );
            convert_mmdit(&src, &mmdit_out);
            cached_ckpt = Some(src);
        });
        run_step(5, 5, &vae_out, force, || {
            let vae_raw = search_dirs
                .iter()
                .flat_map(|d| {
                    [
                        d.join("vae/diffusion_pytorch_model.safetensors"),
                        d.join("diffusion_pytorch_model.safetensors"),
                    ]
                })
                .find(|p| p.exists())
                .or(cached_ckpt)
                .unwrap_or_else(|| {
                    fetch(
                        &client,
                        SD35_OWNER,
                        SD35_NAME,
                        "sd3.5_large_turbo.safetensors",
                        &search_dirs,
                    )
                });
            convert_vae(&vae_raw, &vae_out);
        });
    } else {
        search_dirs.push(PathBuf::from("zimage_turbo_raw"));
        let qwen_out = out_dir.join("qwen3.safetensors");
        let s3dit_out = out_dir.join("s3dit.safetensors");
        let vae_out = out_dir.join("vae.safetensors");
        run_step(1, 3, &qwen_out, force, || {
            let shards = fetch_sharded(
                &client,
                ZIMAGE_OWNER,
                ZIMAGE_NAME,
                "text_encoder",
                "model.safetensors",
                "model.safetensors.index.json",
                &search_dirs,
            );
            convert_qwen3(&shards, &qwen_out);
        });
        run_step(2, 3, &s3dit_out, force, || {
            let shards = fetch_sharded(
                &client,
                ZIMAGE_OWNER,
                ZIMAGE_NAME,
                "transformer",
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.safetensors.index.json",
                &search_dirs,
            );
            convert_s3dit(&shards, &s3dit_out);
        });
        run_step(3, 3, &vae_out, force, || {
            let src = fetch(
                &client,
                ZIMAGE_OWNER,
                ZIMAGE_NAME,
                "vae/diffusion_pytorch_model.safetensors",
                &search_dirs,
            );
            convert_vae(&src, &vae_out);
        });
    }
    println!(
        "Installed {model} weights in {} ({:.2}s)",
        out_dir.display(),
        t_all.elapsed().as_secs_f32()
    );
}
