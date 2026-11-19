#![feature(f16)]

mod engine;
mod install;
mod sd35;
mod vae;
mod zimage;

use engine::get_data_dir;
use sd35::Sd35Pipeline;
use std::env::{args, current_dir, current_exe};
use std::fs::{File, create_dir_all, read_to_string, remove_file, write};
use std::io::{BufRead, BufReader, BufWriter, ErrorKind, Read, Write, stdout};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, exit, id};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::sleep;
use std::time::{Duration, Instant};
use zimage::ZImagePipeline;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Model {
    Sd35,
    ZImage,
}

impl Model {
    fn parse(s: &str) -> Self {
        match s {
            "sd35" => Self::Sd35,
            "zimage" => Self::ZImage,
            other => {
                eprintln!("Error: unknown model `{other}` (expected `sd35` or `zimage`)");
                exit(1);
            }
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Sd35 => "sd35",
            Self::ZImage => "zimage",
        }
    }

    fn weight_dir(self) -> PathBuf {
        get_data_dir().join(self.as_str())
    }

    fn sock_path(self) -> PathBuf {
        get_data_dir().join(format!("{}.sock", self.as_str()))
    }

    fn pid_path(self) -> PathBuf {
        get_data_dir().join(format!("{}.pid", self.as_str()))
    }

    fn packs(self) -> &'static [&'static str] {
        match self {
            Self::Sd35 => &[
                "clip_l.safetensors",
                "clip_g.safetensors",
                "t5xxl.safetensors",
                "mmdit.safetensors",
                "vae.safetensors",
            ],
            Self::ZImage => &["qwen3.safetensors", "s3dit.safetensors", "vae.safetensors"],
        }
    }
}

enum Pipeline {
    Sd35(Sd35Pipeline),
    ZImage(ZImagePipeline),
}

impl Pipeline {
    fn load(model: Model) -> Self {
        let dir = model.weight_dir();
        match model {
            Model::Sd35 => Self::Sd35(Sd35Pipeline::load(&dir)),
            Model::ZImage => Self::ZImage(ZImagePipeline::load(&dir)),
        }
    }

    fn generate(&self, prompt: &str, seed: u64) -> Vec<u8> {
        match self {
            Self::Sd35(p) => p.generate(prompt, seed),
            Self::ZImage(p) => p.generate(prompt, seed),
        }
    }
}

fn write_png_rgb8(path: &Path, w: u32, h: u32, rgb: &[u8]) {
    let file = BufWriter::new(File::create(path).unwrap());
    let mut encoder = png::Encoder::new(file, w, h);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(rgb).unwrap();
}

static STOP_FLAG: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sigterm(_: libc::c_int) {
    STOP_FLAG.store(true, Ordering::SeqCst);
}

fn read_urandom_u64() -> u64 {
    let mut f = File::open("/dev/urandom").unwrap();
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).unwrap();
    u64::from_le_bytes(buf)
}

fn print_usage() {
    let bin = env!("CARGO_PKG_NAME");
    println!(
        "{bin} {}\n{}\n\n\
         Usage:\n  \
           {bin} install <sd35|zimage> [--force] [--raw-dir <DIR>]\n  \
           {bin} start <sd35|zimage>\n  \
           {bin} stop <sd35|zimage>\n  \
           {bin} status [sd35|zimage]\n  \
           {bin} <sd35|zimage> --prompt <TEXT> --path <FILE> [--seed <U64>]\n\n\
         Options:\n  \
           -p, --prompt <TEXT>  Text prompt\n  \
           -o, --path <FILE>    Output PNG path\n  \
           -s, --seed <U64>     RNG seed\n  \
           -h, --help           Show help\n  \
           -V, --version        Show version",
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_DESCRIPTION")
    );
}

fn run_daemon(model: Model) {
    let sock_path = model.sock_path();
    let pid_path = model.pid_path();
    let _ = remove_file(&sock_path);
    unsafe {
        libc::signal(
            libc::SIGTERM,
            handle_sigterm as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGINT,
            handle_sigterm as *const () as libc::sighandler_t,
        );
    }
    let pipeline = Pipeline::load(model);
    let listener = UnixListener::bind(&sock_path).unwrap();
    listener.set_nonblocking(true).unwrap();
    write(&pid_path, id().to_string()).unwrap();
    while !STOP_FLAG.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((mut stream, _)) => {
                if stream.set_nonblocking(false).is_err() {
                    continue;
                }
                let mut cmd = [0u8; 1];
                if stream.read_exact(&mut cmd).is_err() {
                    continue;
                }
                if cmd[0] == b'S' {
                    let msg = format!("Stopped {} daemon.\n", model.as_str());
                    let _ = stream.write_all(msg.as_bytes());
                    break;
                } else if cmd[0] == b'G' {
                    let mut u64_buf = [0u8; 8];
                    let mut u32_buf = [0u8; 4];
                    if stream.read_exact(&mut u64_buf).is_err() {
                        continue;
                    }
                    let init_seed = u64::from_le_bytes(u64_buf);
                    if stream.read_exact(&mut u32_buf).is_err() {
                        continue;
                    }
                    let prompt_len = u32::from_le_bytes(u32_buf) as usize;
                    let mut prompt_bytes = vec![0u8; prompt_len];
                    if stream.read_exact(&mut prompt_bytes).is_err() {
                        continue;
                    }
                    let Ok(prompt) = String::from_utf8(prompt_bytes) else {
                        continue;
                    };
                    if stream.read_exact(&mut u32_buf).is_err() {
                        continue;
                    }
                    let path_len = u32::from_le_bytes(u32_buf) as usize;
                    let mut path_bytes = vec![0u8; path_len];
                    if stream.read_exact(&mut path_bytes).is_err() {
                        continue;
                    }
                    let Ok(path_str) = String::from_utf8(path_bytes) else {
                        continue;
                    };
                    let out_path = PathBuf::from(path_str);
                    let t_e2e = Instant::now();
                    let t_gpu = Instant::now();
                    let rgb_bytes = pipeline.generate(&prompt, init_seed);
                    let gpu_secs = t_gpu.elapsed().as_secs_f32();
                    if let Some(parent) = out_path.parent() {
                        let _ = create_dir_all(parent);
                    }
                    write_png_rgb8(&out_path, 1024, 1024, &rgb_bytes);
                    let msg = format!(
                        "Generated {} ({}, 1024x1024, seed {init_seed}) in {gpu_secs:.2}s (e2e {:.2}s)\n",
                        out_path.display(),
                        model.as_str(),
                        t_e2e.elapsed().as_secs_f32()
                    );
                    let _ = stream.write_all(msg.as_bytes());
                }
            }
            Err(ref e) if e.kind() == ErrorKind::WouldBlock => {
                sleep(Duration::from_millis(20));
            }
            Err(_) => break,
        }
    }
    let _ = remove_file(&sock_path);
    let _ = remove_file(&pid_path);
}

fn require_model_arg(args: &[String], cmd: &str) -> Model {
    let Some(val) = args.first() else {
        eprintln!("Usage: {} {cmd} <sd35|zimage>", env!("CARGO_PKG_NAME"));
        exit(1);
    };
    Model::parse(val)
}

fn run_prompt(model: Model, args: &[String]) {
    let mut prompt: Option<String> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut seed: Option<u64> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-p" | "--prompt" => {
                i += 1;
                prompt = args.get(i).cloned();
            }
            "-o" | "--path" => {
                i += 1;
                out_path = args.get(i).map(PathBuf::from);
            }
            "-s" | "--seed" => {
                i += 1;
                let Some(val) = args.get(i) else {
                    eprintln!("Error: missing value for --seed");
                    exit(1);
                };
                seed = Some(val.parse().unwrap_or_else(|_| {
                    eprintln!("Error: invalid u64 `{val}` for --seed");
                    exit(1);
                }));
            }
            other => {
                eprintln!("Error: unknown argument `{other}`");
                exit(1);
            }
        }
        i += 1;
    }
    let Some(prompt) = prompt else {
        eprintln!("Error: missing required --prompt <TEXT>");
        exit(1);
    };
    let Some(out_path) = out_path else {
        eprintln!("Error: missing required --path <FILE>");
        exit(1);
    };
    if out_path.is_dir() {
        eprintln!("Error: --path must be a file path, not a directory.");
        exit(1);
    }
    let abs_out_path = if out_path.is_absolute() {
        out_path
    } else {
        current_dir().unwrap().join(out_path)
    };
    let seed = seed.unwrap_or_else(read_urandom_u64);
    let sock_path = model.sock_path();
    let mut stream = UnixStream::connect(&sock_path).unwrap_or_else(|_| {
        eprintln!(
            "{} daemon is not running. Start it with: {} start {}",
            model.as_str(),
            env!("CARGO_PKG_NAME"),
            model.as_str(),
        );
        exit(1);
    });
    let prompt_bytes = prompt.as_bytes();
    let path_bytes = abs_out_path.to_str().unwrap().as_bytes();
    stream.write_all(b"G").unwrap();
    stream.write_all(&seed.to_le_bytes()).unwrap();
    stream
        .write_all(&(prompt_bytes.len() as u32).to_le_bytes())
        .unwrap();
    stream.write_all(prompt_bytes).unwrap();
    stream
        .write_all(&(path_bytes.len() as u32).to_le_bytes())
        .unwrap();
    stream.write_all(path_bytes).unwrap();
    stream.flush().unwrap();
    let mut reader = BufReader::new(stream);
    let mut resp = String::new();
    reader.read_line(&mut resp).unwrap();
    print!("{resp}");
}

#[allow(clippy::zombie_processes)]
fn main() {
    let args: Vec<String> = args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        exit(1);
    }
    match args[0].as_str() {
        "-h" | "--help" | "help" => print_usage(),
        "-V" | "--version" => {
            println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        }
        "install" => install::run(&args[1..]),
        "__daemon" => {
            let model = require_model_arg(&args[1..], "__daemon");
            run_daemon(model);
        }
        "status" => {
            let models: &[Model] = if let Some(m) = args.get(1) {
                match Model::parse(m) {
                    Model::Sd35 => &[Model::Sd35],
                    Model::ZImage => &[Model::ZImage],
                }
            } else {
                &[Model::Sd35, Model::ZImage]
            };
            let mut any_running = false;
            for &m in models {
                if UnixStream::connect(m.sock_path()).is_ok() {
                    let pid = read_to_string(m.pid_path()).unwrap_or_default();
                    println!("{:<6} Running (PID {})", m.as_str(), pid.trim());
                    any_running = true;
                } else {
                    println!("{:<6} Stopped", m.as_str());
                }
            }
            if !any_running {
                exit(1);
            }
        }
        "start" => {
            let model = require_model_arg(&args[1..], "start");
            let weight_dir = model.weight_dir();
            for pack in model.packs() {
                if !weight_dir.join(pack).exists() {
                    eprintln!(
                        "Missing {}. Run `{} install {}` first.",
                        weight_dir.join(pack).display(),
                        env!("CARGO_PKG_NAME"),
                        model.as_str()
                    );
                    exit(1);
                }
            }
            let sock_path = model.sock_path();
            let pid_path = model.pid_path();
            if UnixStream::connect(&sock_path).is_ok() {
                println!("{} daemon is already running.", model.as_str());
                return;
            }
            let _ = remove_file(&sock_path);
            let _ = remove_file(&pid_path);
            let t0 = Instant::now();
            let mut child = Command::new(current_exe().unwrap())
                .arg("__daemon")
                .arg(model.as_str())
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap();
            print!(
                "Starting {} daemon (PID {})... ",
                model.as_str(),
                child.id()
            );
            stdout().flush().unwrap();
            while !sock_path.exists() {
                if let Some(status) = child.try_wait().unwrap() {
                    println!("failed");
                    eprintln!("Daemon exited early with status {status}");
                    exit(1);
                }
                sleep(Duration::from_millis(50));
            }
            println!("ready ({:.2}s)", t0.elapsed().as_secs_f32());
        }
        "stop" => {
            let model = require_model_arg(&args[1..], "stop");
            let sock_path = model.sock_path();
            let pid_path = model.pid_path();
            let mut stopped = false;
            if let Ok(mut stream) = UnixStream::connect(&sock_path) {
                let _ = stream.write_all(b"S");
                let mut resp = String::new();
                let _ = stream.read_to_string(&mut resp);
                if !resp.is_empty() {
                    print!("{resp}");
                }
                stopped = true;
            } else if let Ok(pid_str) = read_to_string(&pid_path)
                && let Ok(pid) = pid_str.trim().parse::<i32>()
            {
                unsafe {
                    if libc::kill(pid, libc::SIGTERM) == 0 {
                        println!("Stopped {} daemon (PID {pid}).", model.as_str());
                        stopped = true;
                    }
                }
            }
            let _ = remove_file(&sock_path);
            let _ = remove_file(&pid_path);
            if !stopped {
                eprintln!("{} daemon is not running.", model.as_str());
                exit(1);
            }
        }
        "prompt" => {
            let model = require_model_arg(&args[1..], "prompt");
            run_prompt(model, &args[2..]);
        }
        "sd35" => run_prompt(Model::Sd35, &args[1..]),
        "zimage" => run_prompt(Model::ZImage, &args[1..]),
        other => {
            eprintln!("Error: unknown command or model `{other}`");
            print_usage();
            exit(1);
        }
    }
}
