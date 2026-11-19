use hf_hub::HFClientSync;
use std::{env, fs, path::PathBuf, process::Command};

const TOKENIZERS: &[(&str, &str, &str)] = &[
    ("tokenizer_l.json", "openai", "clip-vit-large-patch14"),
    (
        "tokenizer_g.json",
        "laion",
        "CLIP-ViT-bigG-14-laion2B-39B-b160k",
    ),
    ("tokenizer_t5.json", "google", "flan-t5-xxl"),
    ("tokenizer_qwen3.json", "Qwen", "Qwen3-4B"),
];

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shaders_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let glslc = env::var("GLSLC_PATH").unwrap_or_else(|_| "glslc".into());
    println!("cargo:rerun-if-changed={}", shaders_dir.display());
    println!("cargo:rerun-if-env-changed=GLSLC_PATH");
    for entry in fs::read_dir(&shaders_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "comp") {
            println!("cargo:rerun-if-changed={}", path.display());
            let out = Command::new(&glslc)
                .arg(&path)
                .args(["-O", "--target-env=vulkan1.3", "-o"])
                .arg(
                    out_dir
                        .join(path.file_stem().unwrap())
                        .with_extension("spv"),
                )
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}:\n{}",
                path.display(),
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
    let client = HFClientSync::new().unwrap();
    for &(file, owner, name) in TOKENIZERS {
        let src = client
            .model(owner, name)
            .download_file()
            .filename("tokenizer.json")
            .send()
            .unwrap();
        fs::copy(src, out_dir.join(file)).unwrap();
    }
}
