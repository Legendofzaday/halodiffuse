use hf_hub::HFClientSync;
use rayon::prelude::*;
use safetensors::Dtype;
use safetensors::tensor::{SafeTensors, TensorView, serialize_to_file};
use std::collections::BTreeMap;
use std::fs::{create_dir_all, read, rename};
use std::io::{Write, stdout};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Instant;

const REPO_OWNER: &str = "stabilityai";
const REPO_NAME: &str = "stable-diffusion-3.5-large-turbo";
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
        let view = self.st.tensor(key).unwrap();
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
    assert!(k.is_multiple_of(16) && k.is_multiple_of(16));
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

fn fetch(client: &HFClientSync, filename: &str, raw_dirs: &[PathBuf]) -> PathBuf {
    let basename = Path::new(filename).file_name().unwrap();
    for d in raw_dirs {
        for cand in [d.join(filename), d.join(basename)] {
            if cand.exists() {
                return cand;
            }
        }
    }
    print!("downloading {filename}... ");
    stdout().flush().unwrap();
    client
        .model(REPO_OWNER, REPO_NAME)
        .download_file()
        .filename(filename)
        .send()
        .unwrap()
}

fn run_step(idx: usize, dst: &Path, force: bool, build: impl FnOnce()) {
    let name = dst.file_name().unwrap().to_str().unwrap();
    print!("[{idx}/5] {name:<22} ");
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
    let mut force = false;
    let mut search_dirs = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: {} install [--force] [--raw-dir <DIR> | <DIR>]",
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
            other if !other.starts_with('-') => search_dirs.push(PathBuf::from(other)),
            other => {
                eprintln!("Error: unknown flag `{other}`");
                exit(1);
            }
        }
        i += 1;
    }
    search_dirs.push(PathBuf::from("sd35_turbo_raw"));
    let out_dir = crate::get_data_dir();
    create_dir_all(&out_dir).unwrap();
    let client = HFClientSync::new().unwrap();
    let t_all = Instant::now();
    let clip_l_out = out_dir.join("clip_l.safetensors");
    let clip_g_out = out_dir.join("clip_g.safetensors");
    let t5_out = out_dir.join("t5xxl.safetensors");
    let mmdit_out = out_dir.join("mmdit.safetensors");
    let vae_out = out_dir.join("vae.safetensors");
    let mut cached_ckpt: Option<PathBuf> = None;
    run_step(1, &clip_l_out, force, || {
        let src = fetch(&client, "text_encoders/clip_l.safetensors", &search_dirs);
        convert_clip(&src, &clip_l_out, 768, 12);
    });
    run_step(2, &clip_g_out, force, || {
        let src = fetch(&client, "text_encoders/clip_g.safetensors", &search_dirs);
        convert_clip(&src, &clip_g_out, 1280, 32);
    });
    run_step(3, &t5_out, force, || {
        let src = fetch(
            &client,
            "text_encoders/t5xxl_fp16.safetensors",
            &search_dirs,
        );
        convert_t5(&src, &t5_out);
    });
    run_step(4, &mmdit_out, force, || {
        let src = fetch(&client, "sd3.5_large_turbo.safetensors", &search_dirs);
        convert_mmdit(&src, &mmdit_out);
        cached_ckpt = Some(src);
    });
    run_step(5, &vae_out, force, || {
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
            .unwrap_or_else(|| fetch(&client, "sd3.5_large_turbo.safetensors", &search_dirs));
        convert_vae(&vae_raw, &vae_out);
    });
    println!(
        "Installed weights in {} ({:.2}s)",
        out_dir.display(),
        t_all.elapsed().as_secs_f32()
    );
}
