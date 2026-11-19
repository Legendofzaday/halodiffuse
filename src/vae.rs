use crate::engine::{DeviceScratchArena, Engine, WeightPack};
use std::mem::swap;
use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

pub struct VaeWorkspace {
    x: Subbuffer<[u8]>,
    norm: Subbuffer<[u8]>,
    tmp: Subbuffer<[u8]>,
    qkv_128: Subbuffer<[u8]>,
}

impl VaeWorkspace {
    pub fn new(engine: &Engine, arena: &DeviceScratchArena) -> Self {
        Self {
            x: arena.arena_a.clone(),
            norm: arena.arena_b.clone(),
            tmp: arena.arena_c.clone(),
            qkv_128: engine.alloc_device_bytes(128 * 128 * 1536 * 2),
        }
    }
}

pub fn record_vae_cmd(
    engine: &Engine,
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
        engine
            .bind(
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
        engine
            .bind(
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
        engine
            .bind(
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
        engine
            .bind(
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
    let mut cmd = engine.new_reusable_cmd();
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
            "flash_attention_m16384_h1_d512.comp",
            [1024, 1, 1],
            &[qkv_128, tmp.clone()],
        )
        .record(&mut cmd);
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
    engine
        .bind(
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
