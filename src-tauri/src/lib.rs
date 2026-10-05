pub mod brush;
pub mod colmap;
pub mod frames;
pub mod installer;
pub mod pipeline;
pub mod proc;
pub mod scene;
pub mod tools;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tauri::ipc::Channel;
use tauri::State;
use tokio_util::sync::CancellationToken;

use pipeline::{JobConfig, JobEvent, Layout, ProjectState};
use tools::Tools;

#[derive(Default)]
struct AppState {
    job: Mutex<Option<CancellationToken>>,
    install: Mutex<Option<CancellationToken>>,
}

type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[tauri::command]
fn detect_tools() -> Tools {
    Tools::detect()
}

#[tauri::command]
async fn probe_video(path: String) -> CmdResult<frames::VideoInfo> {
    let ffprobe = Tools::require(&Tools::detect().ffprobe, "ffprobe", tools::install_hint("ffmpeg")).map_err(err)?;
    frames::probe(&ffprobe, &PathBuf::from(path)).await.map_err(|e| format!("{e:#}"))
}

#[tauri::command]
fn default_project_dir(video_path: String) -> String {
    pipeline::default_project_dir(&PathBuf::from(video_path)).to_string_lossy().into_owned()
}

#[tauri::command]
fn load_project(project_dir: String) -> ProjectState {
    Layout::new(project_dir).load()
}

#[tauri::command]
fn scene_frame(project_dir: String) -> CmdResult<scene::SceneFrame> {
    scene::read(&Layout::new(project_dir).model()).map_err(|e| format!("{e:#}"))
}

#[tauri::command]
fn start_job(config: JobConfig, on_event: Channel<JobEvent>, state: State<'_, AppState>) -> CmdResult<()> {
    let mut job = state.job.lock().unwrap();
    if job.as_ref().is_some_and(|t| !t.is_cancelled()) {
        return Err("já existe um processamento em andamento".into());
    }
    let token = CancellationToken::new();
    *job = Some(token.clone());
    let tools = Tools::detect();
    let emit: pipeline::Emit = Arc::new(move |e| {
        let _ = on_event.send(e);
    });
    tauri::async_runtime::spawn(async move {
        pipeline::run(config, tools, emit, token.clone()).await;
        // Marca como encerrado para liberar um novo job.
        token.cancel();
    });
    Ok(())
}

#[tauri::command]
fn cancel_job(state: State<'_, AppState>) {
    if let Some(t) = state.job.lock().unwrap().as_ref() {
        t.cancel();
    }
}

#[tauri::command]
async fn install_plan() -> CmdResult<installer::Plan> {
    installer::plan().await.map_err(|e| format!("{e:#}"))
}

/// O plano é recalculado aqui (URLs e hashes nunca vêm da interface).
#[tauri::command]
async fn install_tools(on_event: Channel<installer::InstallEvent>, state: State<'_, AppState>) -> CmdResult<()> {
    let plan = installer::plan().await.map_err(|e| format!("{e:#}"))?;
    let mut slot = state.install.lock().unwrap();
    if slot.as_ref().is_some_and(|t| !t.is_cancelled()) {
        return Err("a instalação já está em andamento".into());
    }
    let token = CancellationToken::new();
    *slot = Some(token.clone());
    tauri::async_runtime::spawn(async move {
        installer::install(plan, token.clone(), move |e| {
            let _ = on_event.send(e);
        })
        .await;
        token.cancel();
    });
    Ok(())
}

#[tauri::command]
fn cancel_install(state: State<'_, AppState>) {
    if let Some(t) = state.install.lock().unwrap().as_ref() {
        t.cancel();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            detect_tools,
            probe_video,
            default_project_dir,
            load_project,
            scene_frame,
            start_job,
            cancel_job,
            install_plan,
            install_tools,
            cancel_install
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
