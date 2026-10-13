pub mod brush;
pub mod colmap;
pub mod export;
pub mod frames;
pub mod installer;
pub mod masks;
pub mod msg;
pub mod package;
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
    exporting: std::sync::atomic::AtomicBool,
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

/// Loose images and frame folders → sorted list (may scan a large folder on a
/// slow disk, so it runs off the main thread).
#[tauri::command]
async fn probe_images(paths: Vec<String>) -> CmdResult<frames::ImageSetInfo> {
    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    tauri::async_runtime::spawn_blocking(move || frames::probe_images(&paths))
        .await
        .map_err(err)?
        .map_err(|e| format!("{e:#}"))
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
fn set_frames_rejected(project_dir: String, names: Vec<String>, rejected: bool) -> CmdResult<ProjectState> {
    pipeline::set_frames_rejected(&project_dir, &names, rejected).map_err(|e| format!("{e:#}"))
}

#[tauri::command]
fn scene_frame(project_dir: String) -> CmdResult<scene::SceneFrame> {
    scene::read(&Layout::new(project_dir).model()).map_err(|e| format!("{e:#}"))
}

#[tauri::command]
fn splat_info(path: String) -> CmdResult<export::SplatInfo> {
    export::info(&PathBuf::from(path)).map_err(|e| format!("{e:#}"))
}

#[derive(Clone, serde::Serialize)]
struct ExportProgress {
    fraction: f32,
    message: msg::Msg,
}

/// Runs outside the async runtime: it is pure CPU work (rayon) and takes from
/// tenths of a second to ~15 s (SOG with SH degree 3 and 1M gaussians).
#[tauri::command]
async fn export_splat(
    project_dir: String,
    options: export::ExportOptions,
    on_progress: Channel<ExportProgress>,
    state: State<'_, AppState>,
) -> CmdResult<export::ExportStats> {
    if state.job.lock().unwrap().as_ref().is_some_and(|t| !t.is_cancelled()) {
        return Err("wait for processing to finish before exporting".into());
    }
    if state.exporting.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return Err("an export is already in progress".into());
    }
    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut last = std::time::Instant::now() - std::time::Duration::from_secs(1);
        pipeline::export_project(&project_dir, &options, |fraction, message| {
            // ~15 updates per second are enough for the progress bar.
            if last.elapsed().as_millis() >= 66 || fraction >= 1.0 {
                last = std::time::Instant::now();
                let _ = on_progress.send(ExportProgress { fraction, message: message.clone() });
            }
        })
    })
    .await;
    state.exporting.store(false, std::sync::atomic::Ordering::SeqCst);
    result.map_err(err)?.map_err(|e| format!("{e:#}"))
}

/// May run while a job is in progress: the UI only asks for the stages it is
/// not rewriting.
#[tauri::command]
async fn pack_project(request: package::PackRequest, on_progress: Channel<f32>) -> CmdResult<package::PackStats> {
    tauri::async_runtime::spawn_blocking(move || package::pack(&request, throttled(on_progress)))
        .await
        .map_err(err)?
        .map_err(|e| format!("{e:#}"))
}

#[tauri::command]
async fn unpack_project(path: String, on_progress: Channel<f32>) -> CmdResult<package::Unpacked> {
    tauri::async_runtime::spawn_blocking(move || package::unpack(&PathBuf::from(path), throttled(on_progress)))
        .await
        .map_err(err)?
        .map_err(|e| format!("{e:#}"))
}

/// ~15 updates per second are enough for the progress bar.
fn throttled(channel: Channel<f32>) -> impl FnMut(f32) {
    let mut last = std::time::Instant::now() - std::time::Duration::from_secs(1);
    move |fraction| {
        if last.elapsed().as_millis() >= 66 || fraction >= 1.0 {
            last = std::time::Instant::now();
            let _ = channel.send(fraction);
        }
    }
}

#[tauri::command]
fn start_job(config: JobConfig, on_event: Channel<JobEvent>, state: State<'_, AppState>) -> CmdResult<()> {
    let mut job = state.job.lock().unwrap();
    if job.as_ref().is_some_and(|t| !t.is_cancelled()) {
        return Err("processing is already in progress".into());
    }
    if state.exporting.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("wait for the export to finish".into());
    }
    let token = CancellationToken::new();
    *job = Some(token.clone());
    let tools = Tools::detect();
    let emit: pipeline::Emit = Arc::new(move |e| {
        let _ = on_event.send(e);
    });
    tauri::async_runtime::spawn(async move {
        pipeline::run(config, tools, emit, token.clone()).await;
        // Mark as finished so a new job can start.
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
async fn install_plan(rmbg: bool) -> CmdResult<installer::Plan> {
    installer::plan(rmbg).await.map_err(|e| format!("{e:#}"))
}

/// The plan is recomputed here (URLs and hashes never come from the UI).
#[tauri::command]
async fn install_tools(rmbg: bool, on_event: Channel<installer::InstallEvent>, state: State<'_, AppState>) -> CmdResult<()> {
    let plan = installer::plan(rmbg).await.map_err(|e| format!("{e:#}"))?;
    let mut slot = state.install.lock().unwrap();
    if slot.as_ref().is_some_and(|t| !t.is_cancelled()) {
        return Err("installation is already in progress".into());
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
            probe_images,
            default_project_dir,
            load_project,
            set_frames_rejected,
            scene_frame,
            splat_info,
            export_splat,
            pack_project,
            unpack_project,
            start_job,
            cancel_job,
            install_plan,
            install_tools,
            cancel_install
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
