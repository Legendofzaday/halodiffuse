use std::{
    env::var,
    fs::read_dir,
    path::PathBuf,
    process::{Child, Command},
};

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    let collected: Vec<Child> = read_dir("shaders")
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            println!("cargo:rerun-if-changed={}", path.display());
            Command::new("glslc")
                .arg(&path)
                .arg("-o")
                .arg(
                    &PathBuf::from(&var("OUT_DIR").unwrap())
                        .join(path.file_name().unwrap())
                        .with_extension("spv"),
                )
                .args(["-O", "--target-env=vulkan1.3"])
                .spawn()
                .unwrap()
        })
        .collect();
    for mut child in collected {
        assert!(child.wait().unwrap().success());
    }
}
