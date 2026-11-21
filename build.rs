use hf_hub::HFClientSync;
use sha2::{Digest, Sha256};
use std::{
    env,
    fmt::Write,
    fs,
    path::{Path, PathBuf},
    process::Command,
    thread,
};

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

fn sha_hex(h: Sha256) -> String {
    let mut s = String::with_capacity(64);
    for b in h.finalize() {
        write!(s, "{b:02x}").unwrap();
    }
    s
}

fn cache_root(out_dir: &Path) -> PathBuf {
    let base = env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| out_dir.ancestors().nth(5).unwrap().to_path_buf());
    let dir = base.join(".halodiffuse-cache");
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn copy_cached(src: &Path, dst: &Path) -> bool {
    if src.exists() {
        fs::copy(src, dst).unwrap();
        true
    } else {
        false
    }
}

fn extract_linked_objs(mlir_src: &str) -> Vec<String> {
    let mut objs = Vec::new();
    let mut rest = mlir_src;
    while let Some(idx) = rest.find("link_with") {
        rest = &rest[idx + "link_with".len()..];
        let q1 = rest.find('"').unwrap();
        rest = &rest[q1 + 1..];
        let q2 = rest.find('"').unwrap();
        objs.push(rest[..q2].to_string());
        rest = &rest[q2 + 1..];
    }
    objs.sort();
    objs.dedup();
    objs
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shaders_dir = manifest_dir.join("shaders");
    let kernels_dir = manifest_dir.join("kernels");
    let cache_dir = cache_root(&out_dir);
    let glslc = env::var("GLSLC_PATH").unwrap_or("glslc".into());
    let venv = env::var("NPU_VENV").unwrap_or("/home/admin/npu-dev/.venv".into());
    let xrt = env::var("XILINX_XRT").unwrap_or("/opt/xilinx/xrt".into());
    let site = format!("{venv}/lib/python3.12/site-packages");
    let mlir_aie = format!("{site}/mlir_aie");
    let mlir_bin = format!("{mlir_aie}/bin");
    let llvm_bin = format!("{site}/llvm-aie/bin");
    let path_env = format!(
        "{mlir_bin}:{llvm_bin}:{xrt}/bin:{}",
        env::var("PATH").unwrap_or_default()
    );
    let ld_env = format!(
        "{xrt}/lib:{xrt}/lib64:{}",
        env::var("LD_LIBRARY_PATH").unwrap_or_default()
    );
    for dir in [&shaders_dir, &kernels_dir] {
        println!("cargo:rerun-if-changed={}", dir.display());
    }
    for var in ["GLSLC_PATH", "NPU_VENV", "XILINX_XRT"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let shader_flags = ["-O", "--target-env=vulkan1.3"];
    let mut shaders: Vec<PathBuf> = fs::read_dir(&shaders_dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            p.extension().is_some_and(|ext| ext == "comp").then_some(p)
        })
        .collect();
    shaders.sort();
    for p in &shaders {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    let workers = thread::available_parallelism().unwrap().get();
    let chunk_size = shaders.len().div_ceil(workers).max(1);
    thread::scope(|s| {
        for chunk in shaders.chunks(chunk_size) {
            let (glslc, out_dir, cache_dir) = (&glslc, &out_dir, &cache_dir);
            s.spawn(move || {
                for path in chunk {
                    let mut h = Sha256::new();
                    for f in shader_flags {
                        h.update(f);
                    }
                    h.update(fs::read(path).unwrap());
                    let stem = path.file_stem().unwrap().to_str().unwrap();
                    let cached = cache_dir.join(format!("spv_{stem}_{}.spv", sha_hex(h)));
                    let dst = out_dir.join(format!("{stem}.spv"));
                    if copy_cached(&cached, &dst) {
                        continue;
                    }
                    let out = Command::new(glslc)
                        .arg(path)
                        .args(shader_flags)
                        .arg("-o")
                        .arg(&dst)
                        .output()
                        .unwrap();
                    assert!(
                        out.status.success(),
                        "{}:\n{}",
                        path.display(),
                        String::from_utf8_lossy(&out.stderr)
                    );
                    fs::copy(&dst, &cached).unwrap();
                }
            });
        }
    });
    let mut kernel_headers: Vec<PathBuf> = fs::read_dir(&kernels_dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            p.extension()
                .is_some_and(|ext| ext == "h" || ext == "hpp")
                .then_some(p)
        })
        .collect();
    kernel_headers.sort();
    for h in &kernel_headers {
        println!("cargo:rerun-if-changed={}", h.display());
    }
    let inc_aie = format!("-I{mlir_aie}/include");
    let inc_local = format!("-I{}", kernels_dir.display());
    let cc_flags = [
        "--target=aie2p-none-unknown-elf",
        "-std=c++20",
        "-O2",
        "-c",
        "-fno-exceptions",
        "-fno-rtti",
        "-DNDEBUG",
        "-D__AIEARCH__=21",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        &inc_aie,
        &inc_local,
    ];
    for entry in fs::read_dir(&kernels_dir).unwrap() {
        let path = entry.unwrap().path();
        if !path.extension().is_some_and(|e| e == "cc") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let mut h = Sha256::new();
        for f in cc_flags {
            h.update(f);
        }
        for hdr in &kernel_headers {
            h.update(fs::read(hdr).unwrap());
        }
        h.update(fs::read(&path).unwrap());
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let cached = cache_dir.join(format!("aie_{stem}_{}.o", sha_hex(h)));
        let dst = out_dir.join(format!("{stem}.o"));
        if copy_cached(&cached, &dst) {
            continue;
        }
        let out = Command::new(format!("{llvm_bin}/clang++"))
            .args(cc_flags)
            .arg(&path)
            .arg("-o")
            .arg(&dst)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}:\n{}",
            path.display(),
            String::from_utf8_lossy(&out.stderr)
        );
        fs::copy(&dst, &cached).unwrap();
    }
    let mut py_helpers: Vec<PathBuf> = fs::read_dir(&kernels_dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            let is_helper = p.extension().is_some_and(|ext| ext == "py")
                && p.file_name().unwrap().to_str().unwrap().starts_with('_');
            is_helper.then_some(p)
        })
        .collect();
    py_helpers.sort();
    for helper in &py_helpers {
        println!("cargo:rerun-if-changed={}", helper.display());
    }
    for entry in fs::read_dir(&kernels_dir).unwrap() {
        let path = entry.unwrap().path();
        if !path.extension().is_some_and(|e| e == "py") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap();
        if stem.starts_with('_') {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let mut py_h = Sha256::new();
        for helper in &py_helpers {
            py_h.update(fs::read(helper).unwrap());
        }
        py_h.update(fs::read(&path).unwrap());
        let cached_mlir = cache_dir.join(format!("mlir_{stem}_{}.mlir", sha_hex(py_h)));
        let mlir_path = out_dir.join(format!("{stem}.mlir"));
        if !copy_cached(&cached_mlir, &mlir_path) {
            let out = Command::new(format!("{venv}/bin/python3"))
                .env("PATH", &path_env)
                .env("LD_LIBRARY_PATH", &ld_env)
                .env(
                    "PYTHONPATH",
                    format!("{mlir_aie}/python:{}", kernels_dir.display()),
                )
                .env("XILINX_XRT", &xrt)
                .arg(&path)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}:\n{}\n{}",
                path.display(),
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            fs::write(&mlir_path, &out.stdout).unwrap();
            fs::copy(&mlir_path, &cached_mlir).unwrap();
        }
        let mlir_text = fs::read_to_string(&mlir_path).unwrap();
        let mut h = Sha256::new();
        h.update(&mlir_text);
        for obj in extract_linked_objs(&mlir_text) {
            h.update(&obj);
            h.update(fs::read(out_dir.join(&obj)).unwrap());
        }
        let key = sha_hex(h);
        let cached_xclbin = cache_dir.join(format!("npu_{stem}_{key}.xclbin"));
        let cached_insts = cache_dir.join(format!("npu_{stem}_{key}.insts.bin"));
        let xclbin = out_dir.join(format!("{stem}.xclbin"));
        let insts = out_dir.join(format!("{stem}.insts.bin"));
        if cached_xclbin.exists() && cached_insts.exists() {
            fs::copy(&cached_xclbin, &xclbin).unwrap();
            fs::copy(&cached_insts, &insts).unwrap();
            continue;
        }
        let out = Command::new(format!("{venv}/bin/python3"))
            .env("PATH", &path_env)
            .env("LD_LIBRARY_PATH", &ld_env)
            .env("PYTHONPATH", format!("{mlir_aie}/python"))
            .env("XILINX_XRT", &xrt)
            .current_dir(&out_dir)
            .arg(format!("{mlir_bin}/aiecc.py"))
            .args([
                "--aie-generate-xclbin",
                "--aie-generate-npu-insts",
                "--no-compile-host",
                "--no-xchesscc",
                "--no-xbridge",
                &format!("--peano={site}/llvm-aie"),
                &format!("--xclbin-name={}", xclbin.display()),
                &format!("--npu-insts-name={}", insts.display()),
            ])
            .arg(&mlir_path)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}:\n{}\n{}",
            path.display(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        fs::copy(&xclbin, &cached_xclbin).unwrap();
        fs::copy(&insts, &cached_insts).unwrap();
    }
    let shim_src = manifest_dir.join("src/xrt_shim.cc");
    let shim_obj = out_dir.join("xrt_shim.o");
    let shim_lib = out_dir.join("libxrt_shim.a");
    println!("cargo:rerun-if-changed={}", shim_src.display());
    let inc_xrt = format!("-I{xrt}/include");
    let shim_flags = [
        "-std=c++17",
        "-O3",
        "-march=native",
        "-DNDEBUG",
        "-fPIC",
        "-c",
        &inc_xrt,
    ];
    let mut h = Sha256::new();
    for f in shim_flags {
        h.update(f);
    }
    h.update(fs::read(&shim_src).unwrap());
    let cached_shim = cache_dir.join(format!("xrt_shim_{}.a", sha_hex(h)));
    if !copy_cached(&cached_shim, &shim_lib) {
        let out = Command::new("g++")
            .args(shim_flags)
            .arg(&shim_src)
            .arg("-o")
            .arg(&shim_obj)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let out = Command::new("ar")
            .args(["crs"])
            .arg(&shim_lib)
            .arg(&shim_obj)
            .output()
            .unwrap();
        assert!(out.status.success());
        fs::copy(&shim_lib, &cached_shim).unwrap();
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={xrt}/lib");
    println!("cargo:rustc-link-search=native={xrt}/lib64");
    println!("cargo:rustc-link-lib=static=xrt_shim");
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{xrt}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{xrt}/lib64");
    let tok_cache = cache_dir.join("tokenizers");
    fs::create_dir_all(&tok_cache).unwrap();
    let mut client: Option<HFClientSync> = None;
    for &(file, owner, name) in TOKENIZERS {
        let cached = tok_cache.join(file);
        let dst = out_dir.join(file);
        if copy_cached(&cached, &dst) {
            continue;
        }
        let c = client.get_or_insert_with(|| HFClientSync::new().unwrap());
        let src = c
            .model(owner, name)
            .download_file()
            .filename("tokenizer.json")
            .send()
            .unwrap();
        fs::copy(&src, &cached).unwrap();
        fs::copy(&cached, &dst).unwrap();
    }
}
