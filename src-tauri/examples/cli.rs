//! Roda o pipeline sem interface, útil para processar vídeos em lote.
//!
//!   cargo run --release --example cli -- config.json
//!   cargo run --release --example cli -- scene <projeto>   # enquadramento em JSON
//!   cargo run --release --example cli -- export <projeto> [opcoes.json]
//!
//! `config.json` tem o mesmo formato do `JobConfig` enviado pela interface;
//! `opcoes.json`, o do `ExportOptions` (padrão: .sog sem outras reduções).
//! Os eventos saem em stdout, um JSON por linha.

use std::sync::Arc;

use splatforge_lib::pipeline::{self, Emit, JobConfig, JobEvent};
use splatforge_lib::tools::Tools;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("scene") {
        let dir = args.get(1).expect("uso: cli scene <projeto>");
        let frame = splatforge_lib::scene::read(&pipeline::Layout::new(dir).model())?;
        println!("{}", serde_json::to_string(&frame)?);
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("export") {
        let dir = args.get(1).expect("uso: cli export <projeto> [opcoes.json]");
        let opts: splatforge_lib::export::ExportOptions = match args.get(2) {
            Some(p) => serde_json::from_slice(&std::fs::read(p)?)?,
            None => Default::default(),
        };
        let stats = pipeline::export_project(dir, &opts, |f, m| eprintln!("{:>3.0}% {m}", f * 100.0))?;
        println!("{}", serde_json::to_string(&stats)?);
        return Ok(());
    }
    let path = args.first().expect("uso: cli <config.json>");
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
