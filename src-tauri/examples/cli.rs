//! Runs the pipeline headless, useful for batch-processing videos.
//!
//!   cargo run --release --example cli -- config.json
//!   cargo run --release --example cli -- scene <project>   # scene framing as JSON
//!   cargo run --release --example cli -- export <project> [options.json]
//!   cargo run --release --example cli -- pack <project> <dest.gorbit> [video]
//!   cargo run --release --example cli -- unpack <package.gorbit>
//!
//! `config.json` has the same shape as the `JobConfig` sent by the UI;
//! `options.json` that of `ExportOptions` (default: .sog with no other reductions).
//! Events go to stdout, one JSON object per line.

use std::sync::Arc;

use splatforge_lib::pipeline::{self, Emit, JobConfig, JobEvent};
use splatforge_lib::tools::Tools;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("scene") {
        let dir = args.get(1).expect("usage: cli scene <project>");
        let frame = splatforge_lib::scene::read(&pipeline::Layout::new(dir).model())?;
        println!("{}", serde_json::to_string(&frame)?);
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("export") {
        let dir = args.get(1).expect("usage: cli export <project> [options.json]");
        let opts: splatforge_lib::export::ExportOptions = match args.get(2) {
            Some(p) => serde_json::from_slice(&std::fs::read(p)?)?,
            None => Default::default(),
        };
        let stats = pipeline::export_project(dir, &opts, |f, m| eprintln!("{:>3.0}% {m}", f * 100.0))?;
        println!("{}", serde_json::to_string(&stats)?);
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("alpha") {
        // Training dataset with the mask as alpha (edges decontaminated), as the pipeline builds it.
        let (Some(dir), Some(out)) = (args.get(1), args.get(2)) else { panic!("usage: cli alpha <project> <out_dir>") };
        let layout = pipeline::Layout::new(dir);
        let t = std::time::Instant::now();
        splatforge_lib::masks::build_alpha_dataset(&layout.dataset(), &layout.masks(), std::path::Path::new(out), None)?;
        eprintln!("alpha dataset in {:.1}s", t.elapsed().as_secs_f32());
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("pack") {
        use splatforge_lib::package::{pack, Include, PackRequest, Source};
        let (Some(dir), Some(dest)) = (args.get(1), args.get(2)) else { panic!("usage: cli pack <project> <dest.gorbit> [video]") };
        let req = PackRequest {
            project_dir: dir.clone(),
            dest: dest.clone(),
            include: Include { masks: true, poses: true, splat: true },
            source: args.get(3).map(|p| Source::Video { path: p.clone() }),
            session: None,
        };
        let stats = pack(&req, |f| eprint!("\r{:>3.0}%", f * 100.0))?;
        println!("\n{}", serde_json::to_string(&stats)?);
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("unpack") {
        let pkg = args.get(1).expect("usage: cli unpack <package.gorbit>");
        let r = splatforge_lib::package::unpack(std::path::Path::new(pkg), |f| eprint!("\r{:>3.0}%", f * 100.0))?;
        println!("\n{}", serde_json::to_string(&r)?);
        return Ok(());
    }
    let path = args.first().expect("usage: cli <config.json>");
    let cfg: JobConfig = serde_json::from_slice(&std::fs::read(path)?)?;
    let tools = Tools::detect();
    eprintln!("{}", serde_json::to_string(&tools)?);

    let failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = failed.clone();
    let emit: Emit = Arc::new(move |e| {
        if matches!(e, JobEvent::Failed { .. }) {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        println!("{}", serde_json::to_string(&e).unwrap());
    });

    let cancel = CancellationToken::new();
    let ctrl_c = cancel.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        ctrl_c.cancel();
    });

    pipeline::run(cfg, tools, emit, cancel).await;
    if failed.load(std::sync::atomic::Ordering::Relaxed) {
        std::process::exit(1);
    }
    Ok(())
}
