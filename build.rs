use std::{
    env::var,
    fs::read_dir,
    path::PathBuf,
    process::{Child, Command},
};

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    let out_dir = PathBuf::from(var("OUT_DIR").unwrap());
    let collected: Vec<Child> = read_dir("shaders")
        .unwrap()
        .filter_map(|entry| {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) != Some("comp") {
                return None;
            }
            println!("cargo:rerun-if-changed={}", path.display());
            Some(
                Command::new("glslc")
                    .arg(&path)
                    .arg("-o")
                    .arg(
                        out_dir
                            .join(path.file_name().unwrap())
                            .with_extension("spv"),
                    )
                    .args(["-O", "--target-env=vulkan1.3"])
                    .spawn()
                    .unwrap(),
            )
        })
        .collect();
    for mut child in collected {
        assert!(child.wait().unwrap().success());
    }
    for (file, url) in [
        (
            "tokenizer_l.json",
            "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json",
        ),
        (
            "tokenizer_g.json",
            "https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/tokenizer.json",
        ),
        (
            "tokenizer_t5.json",
            "https://huggingface.co/google/t5-v1_1-xxl/resolve/main/tokenizer.json",
        ),
    ] {
        let dst = out_dir.join(file);
        if !dst.exists() {
            assert!(
                Command::new("curl")
                    .args(["-sSL", url, "-o"])
                    .arg(&dst)
                    .status()
                    .unwrap()
                    .success()
            );
        }
    }
}
