import os

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

import argparse
import time
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

REPO_ID = "stabilityai/stable-diffusion-3.5-large-turbo"
OUT_DIR = Path.home() / ".local/share/halodiffuse"
PERM_N = torch.tensor([((r & 1) << 3) | (r >> 1) for r in range(16)], dtype=torch.long)

def to_fp16_contig(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32).clamp(-65504.0, 65504.0).to(torch.float16).contiguous()

def pack_b_tiles_16x16(b_logical: torch.Tensor) -> torch.Tensor:
    b = to_fp16_contig(b_logical)
    k, n = b.shape
    t = b.reshape(k // 16, 16, n // 16, 16)[:, :, :, PERM_N]
    return t.permute(0, 3, 2, 1).reshape(k, n).contiguous()

def strip_prefix(state_dict: dict, prefixes: list[str]) -> dict:
    out = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
                break
        out[new_k] = v
    return out

def rmsnorm_fp16(x_f16: torch.Tensor, gamma_f16: torch.Tensor) -> torch.Tensor:
    x = x_f16.float()
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return (x * rstd * gamma_f16.float()).to(torch.float16)

def gelu_new_fp32(x: torch.Tensor) -> torch.Tensor:
    c = -2.30198566 * (x + 0.044715 * x * x * x)
    return x / (1.0 + torch.exp2(c))

def fold_adaln_bias_6(bias: torch.Tensor) -> torch.Tensor:
    b = bias.to(torch.float32).clone()
    b[2432:4864] += 1.0
    b[9728:12160] += 1.0
    return b

def fold_adaln_bias_2(bias: torch.Tensor) -> torch.Tensor:
    b = bias.to(torch.float32).clone()
    b[2432:4864] += 1.0
    return b

def convert_clip(input_path: Path, output_path: Path, d_model: int, num_layers: int):
    print(f"[*] Packing CLIP (d={d_model}) -> {output_path}...")
    sd = strip_prefix(
        load_file(str(input_path)),
        ["text_encoders.clip_l.transformer.", "text_encoders.clip_g.transformer.", ""],
    )
    out = {
        "embeddings.token_embedding.weight": to_fp16_contig(sd["text_model.embeddings.token_embedding.weight"]),
        "embeddings.position_embedding.weight": to_fp16_contig(sd["text_model.embeddings.position_embedding.weight"]),
    }
    for i in range(num_layers):
        p = f"text_model.encoder.layers.{i}"
        out[f"layers.{i}.ln1.gamma"] = to_fp16_contig(sd[f"{p}.layer_norm1.weight"])
        out[f"layers.{i}.ln1.beta"] = to_fp16_contig(sd[f"{p}.layer_norm1.bias"])
        out[f"layers.{i}.ln2.gamma"] = to_fp16_contig(sd[f"{p}.layer_norm2.weight"])
        out[f"layers.{i}.ln2.beta"] = to_fp16_contig(sd[f"{p}.layer_norm2.bias"])
        wq, wk, wv = sd[f"{p}.self_attn.q_proj.weight"], sd[f"{p}.self_attn.k_proj.weight"], sd[f"{p}.self_attn.v_proj.weight"]
        bq, bk, bv = sd[f"{p}.self_attn.q_proj.bias"], sd[f"{p}.self_attn.k_proj.bias"], sd[f"{p}.self_attn.v_proj.bias"]
        out[f"layers.{i}.attn.qkv.weight"] = pack_b_tiles_16x16(torch.cat([wq, wk, wv], dim=0).T)
        out[f"layers.{i}.attn.qkv.bias"] = to_fp16_contig(torch.cat([bq, bk, bv], dim=0))
        out[f"layers.{i}.attn.out_proj.weight"] = pack_b_tiles_16x16(sd[f"{p}.self_attn.out_proj.weight"].T)
        out[f"layers.{i}.attn.out_proj.bias"] = to_fp16_contig(sd[f"{p}.self_attn.out_proj.bias"])
        out[f"layers.{i}.mlp.fc1.weight"] = pack_b_tiles_16x16(sd[f"{p}.mlp.fc1.weight"].T)
        out[f"layers.{i}.mlp.fc1.bias"] = to_fp16_contig(sd[f"{p}.mlp.fc1.bias"])
        out[f"layers.{i}.mlp.fc2.weight"] = pack_b_tiles_16x16(sd[f"{p}.mlp.fc2.weight"].T)
        out[f"layers.{i}.mlp.fc2.bias"] = to_fp16_contig(sd[f"{p}.mlp.fc2.bias"])
    out["final_ln.gamma"] = to_fp16_contig(sd["text_model.final_layer_norm.weight"])
    out["final_ln.beta"] = to_fp16_contig(sd["text_model.final_layer_norm.bias"])
    proj_w = sd["text_projection.weight"].T if "text_projection.weight" in sd else sd.get("text_projection", torch.eye(d_model))
    out["eos_proj.weight"] = to_fp16_contig(proj_w)
    out["eos_proj.bias"] = to_fp16_contig(sd.get("text_projection.bias", torch.zeros(d_model)))
    save_file(out, str(output_path))

@torch.no_grad()
def convert_t5(input_path: Path, output_path: Path):
    print(f"[*] Calibrating & Packing T5-XXL -> {output_path}...")
    sd = load_file(str(input_path))
    emb_key = "shared.weight" if "shared.weight" in sd else "encoder.embed_tokens.weight"
    rpb_key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    res_scale = 32.0
    emb_w = (sd[emb_key].float() / res_scale).to(torch.float16)
    out = {
        "embed_tokens.weight": to_fp16_contig(emb_w),
        "rpb_weight": to_fp16_contig(sd[rpb_key]),
    }
    tokens = torch.zeros(256, dtype=torch.long)
    tokens[0] = 1
    x = emb_w[tokens]
    for i in range(24):
        p = f"encoder.block.{i}"
        ln1_g = sd[f"{p}.layer.0.layer_norm.weight"].float()
        wq = sd[f"{p}.layer.0.SelfAttention.q.weight"].float()
        wk = sd[f"{p}.layer.0.SelfAttention.k.weight"].float()
        wv = sd[f"{p}.layer.0.SelfAttention.v.weight"].float()
        wo_attn = sd[f"{p}.layer.0.SelfAttention.o.weight"].float() / res_scale
        norm1_test = rmsnorm_fp16(x, ln1_g.to(torch.float16))
        if torch.isinf(norm1_test).any() or norm1_test.float().abs().max().item() > 8000.0:
            ln1_g = ln1_g / 8.0
            wq, wk, wv = wq * 8.0, wk * 8.0, wv * 8.0
        ln1_g_f16 = to_fp16_contig(ln1_g)
        norm1 = rmsnorm_fp16(x, ln1_g_f16)
        qkv_f32 = norm1.float() @ torch.cat([wq, wk, wv], dim=0).T
        if qkv_f32[:, 8192:].abs().max().item() > 16000.0:
            wv = wv / 8.0
            wo_attn = wo_attn * 8.0
        out[f"blocks.{i}.ln1.gamma"] = ln1_g_f16
        out[f"blocks.{i}.attn.qkv.weight"] = pack_b_tiles_16x16(torch.cat([wq, wk, wv], dim=0).T)
        out[f"blocks.{i}.attn.o.weight"] = pack_b_tiles_16x16(wo_attn.T)
        v_f16 = (norm1.float() @ wv.T).to(torch.float16)
        x = (x.float() + v_f16.float() @ wo_attn.T).to(torch.float16)

        ln2_g = sd[f"{p}.layer.1.layer_norm.weight"].float()
        wi_0 = sd[f"{p}.layer.1.DenseReluDense.wi_0.weight"].float()
        wi_1 = sd[f"{p}.layer.1.DenseReluDense.wi_1.weight"].float()
        wo_ffn = sd[f"{p}.layer.1.DenseReluDense.wo.weight"].float() / res_scale
        norm2_test = rmsnorm_fp16(x, ln2_g.to(torch.float16))
        if torch.isinf(norm2_test).any() or norm2_test.float().abs().max().item() > 8000.0:
            ln2_g = ln2_g / 8.0
            wi_0, wi_1 = wi_0 * 8.0, wi_1 * 8.0
        ln2_g_f16 = to_fp16_contig(ln2_g)
        norm2 = rmsnorm_fp16(x, ln2_g_f16)
        u = gelu_new_fp32(norm2.float() @ wi_0.T)
        g = norm2.float() @ wi_1.T
        ffn_raw_max = (u * g).abs().max().item()
        wo_max = wo_ffn.abs().max().item()
        ffn_div = 1.0
        while (ffn_raw_max / ffn_div > 8000.0) and (wo_max * ffn_div * 2.0 < 30000.0):
            ffn_div *= 2.0
        wi_1_scaled = wi_1 / ffn_div
        wo_ffn_scaled = wo_ffn * ffn_div
        ffn_hid_f16 = (u * (norm2.float() @ wi_1_scaled.T)).clamp(-65504.0, 65504.0).to(torch.float16)
        x = (x.float() + ffn_hid_f16.float() @ wo_ffn_scaled.T).clamp(-65504.0, 65504.0).to(torch.float16)
        out[f"blocks.{i}.ln2.gamma"] = ln2_g_f16
        out[f"blocks.{i}.ffn.b0_wi0.weight"] = pack_b_tiles_16x16(wi_0.T)
        out[f"blocks.{i}.ffn.b1_wi1.weight"] = pack_b_tiles_16x16(wi_1_scaled.T)
        out[f"blocks.{i}.ffn.wo.weight"] = pack_b_tiles_16x16(wo_ffn_scaled.T)
    out["final_ln.gamma"] = to_fp16_contig(sd["encoder.final_layer_norm.weight"])
    save_file(out, str(output_path))

def convert_mmdit(input_path: Path, output_path: Path):
    print(f"[*] Packing MMDiT -> {output_path}...")
    raw_sd = load_file(str(input_path))
    sd = {
        (k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k): v
        for k, v in raw_sd.items()
        if not k.startswith("first_stage_model.") and not k.startswith("text_encoders.")
    }
    out = {
        "t_embedder.mlp.0.weight": to_fp16_contig(sd["t_embedder.mlp.0.weight"].T),
        "t_embedder.mlp.0.bias": to_fp16_contig(sd["t_embedder.mlp.0.bias"]),
        "t_embedder.mlp.2.weight": to_fp16_contig(sd["t_embedder.mlp.2.weight"].T),
        "t_embedder.mlp.2.bias": to_fp16_contig(sd["t_embedder.mlp.2.bias"]),
        "y_embedder.mlp.0.weight": to_fp16_contig(sd["y_embedder.mlp.0.weight"].T),
        "y_embedder.mlp.0.bias": to_fp16_contig(sd["y_embedder.mlp.0.bias"]),
        "y_embedder.mlp.2.weight": to_fp16_contig(sd["y_embedder.mlp.2.weight"].T),
        "y_embedder.mlp.2.bias": to_fp16_contig(sd["y_embedder.mlp.2.bias"]),
        "x_embedder.proj.weight": pack_b_tiles_16x16(sd["x_embedder.proj.weight"].reshape(2432, 64).T),
        "context_embedder.weight": pack_b_tiles_16x16(sd["context_embedder.weight"].T),
        "context_embedder.bias": to_fp16_contig(sd["context_embedder.bias"]),
    }
    pos_embed = sd["pos_embed"].to(torch.float32)
    grid_size = int(pos_embed.shape[1] ** 0.5)
    top = (grid_size - 64) // 2
    pos_crop = pos_embed.reshape(1, grid_size, grid_size, 2432)[:, top : top + 64, top : top + 64, :].reshape(4096, 2432)
    out["x_embedder.fused_pos_bias"] = to_fp16_contig(pos_crop + sd["x_embedder.proj.bias"].to(torch.float32).unsqueeze(0))
    for i in range(38):
        px, pc = f"joint_blocks.{i}.x_block", f"joint_blocks.{i}.context_block"
        is_last = i == 37
        wx_ada = sd[f"{px}.adaLN_modulation.1.weight"].T
        bx_ada = fold_adaln_bias_6(sd[f"{px}.adaLN_modulation.1.bias"])
        if not is_last:
            wc_ada = sd[f"{pc}.adaLN_modulation.1.weight"].T
            bc_ada = fold_adaln_bias_6(sd[f"{pc}.adaLN_modulation.1.bias"])
            out[f"blocks.{i}.adaln_14592.weight"] = to_fp16_contig(torch.stack([wx_ada, wc_ada], dim=0))
            out[f"blocks.{i}.adaln_14592.bias"] = to_fp16_contig(torch.stack([bx_ada, bc_ada], dim=0))
        else:
            out[f"blocks.{i}.x_adaln_14592.weight"] = to_fp16_contig(wx_ada)
            out[f"blocks.{i}.x_adaln_14592.bias"] = to_fp16_contig(bx_ada)
            out[f"blocks.{i}.ctx_adaln_4864.weight"] = to_fp16_contig(sd[f"{pc}.adaLN_modulation.1.weight"].T)
            out[f"blocks.{i}.ctx_adaln_4864.bias"] = to_fp16_contig(fold_adaln_bias_2(sd[f"{pc}.adaLN_modulation.1.bias"]))
        out[f"blocks.{i}.x_attn.qkv.weight"] = pack_b_tiles_16x16(sd[f"{px}.attn.qkv.weight"].T)
        out[f"blocks.{i}.x_attn.qkv.bias"] = to_fp16_contig(sd[f"{px}.attn.qkv.bias"])
        out[f"blocks.{i}.ctx_attn.qkv.weight"] = pack_b_tiles_16x16(sd[f"{pc}.attn.qkv.weight"].T)
        out[f"blocks.{i}.ctx_attn.qkv.bias"] = to_fp16_contig(sd[f"{pc}.attn.qkv.bias"])
        out[f"blocks.{i}.qk_norm.gamma_x_q"] = to_fp16_contig(sd[f"{px}.attn.ln_q.weight"])
        out[f"blocks.{i}.qk_norm.gamma_x_k"] = to_fp16_contig(sd[f"{px}.attn.ln_k.weight"])
        out[f"blocks.{i}.qk_norm.gamma_ctx_q"] = to_fp16_contig(sd[f"{pc}.attn.ln_q.weight"])
        out[f"blocks.{i}.qk_norm.gamma_ctx_k"] = to_fp16_contig(sd[f"{pc}.attn.ln_k.weight"])
        out[f"blocks.{i}.x_attn.proj.weight"] = pack_b_tiles_16x16(sd[f"{px}.attn.proj.weight"].T)
        out[f"blocks.{i}.x_attn.proj.bias"] = to_fp16_contig(sd[f"{px}.attn.proj.bias"])
        out[f"blocks.{i}.x_mlp.fc1.weight"] = pack_b_tiles_16x16(sd[f"{px}.mlp.fc1.weight"].T)
        out[f"blocks.{i}.x_mlp.fc1.bias"] = to_fp16_contig(sd[f"{px}.mlp.fc1.bias"])
        out[f"blocks.{i}.x_mlp.fc2.weight"] = pack_b_tiles_16x16(sd[f"{px}.mlp.fc2.weight"].T)
        out[f"blocks.{i}.x_mlp.fc2.bias"] = to_fp16_contig(sd[f"{px}.mlp.fc2.bias"])
        if not is_last:
            out[f"blocks.{i}.ctx_attn.proj.weight"] = pack_b_tiles_16x16(sd[f"{pc}.attn.proj.weight"].T)
            out[f"blocks.{i}.ctx_attn.proj.bias"] = to_fp16_contig(sd[f"{pc}.attn.proj.bias"])
            out[f"blocks.{i}.ctx_mlp.fc1.weight"] = pack_b_tiles_16x16(sd[f"{pc}.mlp.fc1.weight"].T)
            out[f"blocks.{i}.ctx_mlp.fc1.bias"] = to_fp16_contig(sd[f"{pc}.mlp.fc1.bias"])
            out[f"blocks.{i}.ctx_mlp.fc2.weight"] = pack_b_tiles_16x16(sd[f"{pc}.mlp.fc2.weight"].T)
            out[f"blocks.{i}.ctx_mlp.fc2.bias"] = to_fp16_contig(sd[f"{pc}.mlp.fc2.bias"])
    out["final_layer.adaln_4864.weight"] = to_fp16_contig(sd["final_layer.adaLN_modulation.1.weight"].T)
    out["final_layer.adaln_4864.bias"] = to_fp16_contig(fold_adaln_bias_2(sd["final_layer.adaLN_modulation.1.bias"]))
    fl_w = sd["final_layer.linear.weight"].reshape(2, 2, 16, 2432).permute(2, 0, 1, 3).reshape(64, 2432).T
    fl_b = sd["final_layer.linear.bias"].reshape(2, 2, 16).permute(2, 0, 1).reshape(64)
    out["final_layer.linear.weight"] = pack_b_tiles_16x16(fl_w)
    out["final_layer.linear.bias"] = to_fp16_contig(fl_b)
    save_file(out, str(output_path))

def convert_vae(input_path: Path, output_path: Path):
    print(f"[*] Packing VAE -> {output_path}...")
    sd = strip_prefix(load_file(str(input_path)), ["first_stage_model.", ""])
    is_diffusers = "decoder.up_blocks.0.resnets.0.conv1.weight" in sd
    out = {}

    def get_wb(diff_key: str, ldm_key: str):
        base = diff_key if is_diffusers else ldm_key
        return sd[f"{base}.weight"], sd[f"{base}.bias"]

    def add_conv3x3(name: str, diff_key: str, ldm_key: str):
        w, b = get_wb(diff_key, ldm_key)
        out[f"{name}.weight"] = pack_b_tiles_16x16(w.permute(2, 3, 1, 0).reshape(-1, w.shape[0]))
        out[f"{name}.bias"] = to_fp16_contig(b)

    def add_conv1x1(name: str, diff_key: str, ldm_key: str):
        w, b = get_wb(diff_key, ldm_key)
        out[f"{name}.weight"] = pack_b_tiles_16x16(w.squeeze(-1).squeeze(-1).T)
        out[f"{name}.bias"] = to_fp16_contig(b)

    def add_gn(name: str, diff_key: str, ldm_key: str):
        w, b = get_wb(diff_key, ldm_key)
        out[f"{name}.gamma"] = to_fp16_contig(w)
        out[f"{name}.beta"] = to_fp16_contig(b)

    add_conv3x3("conv_in", "decoder.conv_in", "decoder.conv_in")
    add_gn("mid.block_1.norm1", "decoder.mid_block.resnets.0.norm1", "decoder.mid.block_1.norm1")
    add_conv3x3("mid.block_1.conv1", "decoder.mid_block.resnets.0.conv1", "decoder.mid.block_1.conv1")
    add_gn("mid.block_1.norm2", "decoder.mid_block.resnets.0.norm2", "decoder.mid.block_1.norm2")
    add_conv3x3("mid.block_1.conv2", "decoder.mid_block.resnets.0.conv2", "decoder.mid.block_1.conv2")
    add_gn("mid.attn.norm", "decoder.mid_block.attentions.0.group_norm", "decoder.mid.attn_1.norm")
    wq, bq = get_wb("decoder.mid_block.attentions.0.to_q", "decoder.mid.attn_1.q")
    wk, bk = get_wb("decoder.mid_block.attentions.0.to_k", "decoder.mid.attn_1.k")
    wv, bv = get_wb("decoder.mid_block.attentions.0.to_v", "decoder.mid.attn_1.v")
    out["mid.attn.qkv.weight"] = pack_b_tiles_16x16(torch.cat([wq, wk, wv], dim=0).reshape(1536, 512).T)
    out["mid.attn.qkv.bias"] = to_fp16_contig(torch.cat([bq, bk, bv], dim=0))
    add_conv1x1("mid.attn.proj_out", "decoder.mid_block.attentions.0.to_out.0", "decoder.mid.attn_1.proj_out")
    add_gn("mid.block_2.norm1", "decoder.mid_block.resnets.1.norm1", "decoder.mid.block_2.norm1")
    add_conv3x3("mid.block_2.conv1", "decoder.mid_block.resnets.1.conv1", "decoder.mid.block_2.conv1")
    add_gn("mid.block_2.norm2", "decoder.mid_block.resnets.1.norm2", "decoder.mid.block_2.norm2")
    add_conv3x3("mid.block_2.conv2", "decoder.mid_block.resnets.1.conv2", "decoder.mid.block_2.conv2")
    for up_idx in range(4):
        ldm_up_idx = 3 - up_idx
        for res_idx in range(3):
            p_diff = f"decoder.up_blocks.{up_idx}.resnets.{res_idx}"
            p_ldm = f"decoder.up.{ldm_up_idx}.block.{res_idx}"
            prefix = f"up.{up_idx}.res.{res_idx}"
            add_gn(f"{prefix}.norm1", f"{p_diff}.norm1", f"{p_ldm}.norm1")
            add_conv3x3(f"{prefix}.conv1", f"{p_diff}.conv1", f"{p_ldm}.conv1")
            add_gn(f"{prefix}.norm2", f"{p_diff}.norm2", f"{p_ldm}.norm2")
            add_conv3x3(f"{prefix}.conv2", f"{p_diff}.conv2", f"{p_ldm}.conv2")
            if up_idx in (2, 3) and res_idx == 0:
                add_conv1x1(f"{prefix}.conv_shortcut", f"{p_diff}.conv_shortcut", f"{p_ldm}.nin_shortcut")
        if up_idx < 3:
            add_conv3x3(f"up.{up_idx}.upsample_conv", f"decoder.up_blocks.{up_idx}.upsamplers.0.conv", f"decoder.up.{ldm_up_idx}.upsample.conv")
    add_gn("norm_out", "decoder.conv_norm_out", "decoder.norm_out")
    w_out, b_out = get_wb("decoder.conv_out", "decoder.conv_out")
    out["conv_out.weight"] = to_fp16_contig((w_out.to(torch.float32) * 127.5).permute(0, 2, 3, 1))
    out["conv_out.bias"] = to_fp16_contig(b_out.to(torch.float32) * 127.5 + 127.5)
    save_file(out, str(output_path))

def fetch(filename: str, raw_dirs: list[Path]) -> Path:
    basename = Path(filename).name
    for d in raw_dirs:
        for candidate in (d / filename, d / basename):
            if candidate.exists():
                print(f"[*] Using local weight file: {candidate}")
                return candidate

    for attempt in range(1, 11):
        try:
            print(f"[*] Downloading {filename} from {REPO_ID} (attempt {attempt}/10, HTTP Range resume enabled)...")
            return Path(hf_hub_download(repo_id=REPO_ID, filename=filename))
        except Exception as e:
            print(f"[!] Download interrupted ({e}). Retrying in 5 seconds...")
            time.sleep(5)
    raise RuntimeError(f"Failed to download {filename} after 10 attempts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and pack SD 3.5 Large Turbo weights for HaloDiffuse")
    parser.add_argument("path", nargs="?", type=Path, default=None, help="Optional path to folder with already-downloaded weights")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Optional path to folder with already-downloaded weights")
    parser.add_argument("--force", action="store_true", help="Re-pack weights even if output files already exist")
    args = parser.parse_args()

    search_dirs = []
    if args.path is not None:
        search_dirs.append(args.path.expanduser().resolve())
    if args.raw_dir is not None:
        search_dirs.append(args.raw_dir.expanduser().resolve())
    search_dirs.append(Path("sd35_turbo_raw").resolve())

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clip_l_out = OUT_DIR / "clip_l.safetensors"
    if args.force or not clip_l_out.exists():
        convert_clip(fetch("text_encoders/clip_l.safetensors", search_dirs), clip_l_out, 768, 12)
    else:
        print(f"[✓] Skipping {clip_l_out} (already exists)")

    clip_g_out = OUT_DIR / "clip_g.safetensors"
    if args.force or not clip_g_out.exists():
        convert_clip(fetch("text_encoders/clip_g.safetensors", search_dirs), clip_g_out, 1280, 32)
    else:
        print(f"[✓] Skipping {clip_g_out} (already exists)")

    t5_out = OUT_DIR / "t5xxl.safetensors"
    if args.force or not t5_out.exists():
        convert_t5(fetch("text_encoders/t5xxl_fp16.safetensors", search_dirs), t5_out)
    else:
        print(f"[✓] Skipping {t5_out} (already exists)")

    mmdit_out = OUT_DIR / "mmdit.safetensors"
    vae_out = OUT_DIR / "vae.safetensors"
    if args.force or not mmdit_out.exists() or not vae_out.exists():
        mmdit_raw = fetch("sd3.5_large_turbo.safetensors", search_dirs)
        if args.force or not mmdit_out.exists():
            convert_mmdit(mmdit_raw, mmdit_out)
        else:
            print(f"[✓] Skipping {mmdit_out} (already exists)")

        if args.force or not vae_out.exists():
            vae_raw = mmdit_raw
            for d in search_dirs:
                for cand in (d / "vae/diffusion_pytorch_model.safetensors", d / "diffusion_pytorch_model.safetensors"):
                    if cand.exists():
                        vae_raw = cand
                        break
            convert_vae(vae_raw, vae_out)
        else:
            print(f"[✓] Skipping {vae_out} (already exists)")
    else:
        print(f"[✓] Skipping {mmdit_out} and {vae_out} (already exist)")

    print(f"[+] All weights ready in {OUT_DIR}!")