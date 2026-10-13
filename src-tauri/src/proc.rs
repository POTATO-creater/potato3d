//! Runs external processes (ffmpeg, colmap, brush) with line-by-line output
//! streaming and cooperative cancellation.

use std::collections::VecDeque;
use std::process::Stdio;

use anyhow::{bail, Context, Result};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Sentinel error to tell "user cancelled" apart from a real failure.
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

pub fn is_cancelled(err: &anyhow::Error) -> bool {
    err.downcast_ref::<Cancelled>().is_some()
}

/// On Windows, a GUI app that launches console programs opens a black window
/// for each one. `CREATE_NO_WINDOW` prevents that.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

pub fn hide_console(cmd: &mut Command) {
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    #[cfg(not(windows))]
    let _ = cmd;
}

pub fn hide_console_std(cmd: &mut std::process::Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = cmd;
}

/// Reads a stream splitting on `\n` **and** `\r`, to catch progress bars
/// that keep rewriting the same line.
async fn pump_lines<R: AsyncRead + Unpin>(mut reader: R, tx: mpsc::UnboundedSender<String>) {
    let mut buf = vec![0u8; 8192];
    let mut pending: Vec<u8> = Vec::new();
    loop {
        let n = match reader.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        for &b in &buf[..n] {
            if b == b'\n' || b == b'\r' {
                if !pending.is_empty() {
                    let line = String::from_utf8_lossy(&pending).trim_end().to_string();
                    pending.clear();
                    if !line.is_empty() && tx.send(line).is_err() {
                        return;
                    }
                }
            } else {
                pending.push(b);
            }
        }
    }
    if !pending.is_empty() {
        let _ = tx.send(String::from_utf8_lossy(&pending).trim_end().to_string());
    }
}

/// Runs `cmd` to completion, calling `on_line` for each stdout/stderr line.
/// Kills the process if `cancel` fires. On failure, the error includes the
/// last output lines for diagnosis.
pub async fn run_streaming(
    mut cmd: Command,
    cancel: &CancellationToken,
    mut on_line: impl FnMut(&str),
) -> Result<()> {
    let program = cmd.as_std().get_program().to_string_lossy().into_owned();
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    hide_console(&mut cmd);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("could not run {program}"))?;

    let (tx, mut rx) = mpsc::unbounded_channel();
    tokio::spawn(pump_lines(child.stdout.take().unwrap(), tx.clone()));
    tokio::spawn(pump_lines(child.stderr.take().unwrap(), tx));

    let mut tail: VecDeque<String> = VecDeque::with_capacity(25);
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                let _ = child.kill().await;
                return Err(Cancelled.into());
            }
            line = rx.recv() => match line {
                Some(line) => {
                    on_line(&line);
                    if tail.len() == 25 {
                        tail.pop_front();
                    }
                    tail.push_back(line);
                }
                None => break,
            }
        }
    }

    let status = tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            let _ = child.kill().await;
            return Err(Cancelled.into());
        }
        status = child.wait() => status?,
    };

    if !status.success() {
        let name = std::path::Path::new(&program)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or(program);
        bail!(
            "{name} exited with code {}\n{}",
            status.code().map_or("?".into(), |c| c.to_string()),
            tail.into_iter().collect::<Vec<_>>().join("\n")
        );
    }
    Ok(())
}

/// Runs and captures the full output (for short commands such as ffprobe).
pub async fn run_capture(mut cmd: Command) -> Result<String> {
    let program = cmd.as_std().get_program().to_string_lossy().into_owned();
    hide_console(&mut cmd);
    let out = cmd
        .stdin(Stdio::null())
        .output()
        .await
        .with_context(|| format!("could not run {program}"))?;
    if !out.status.success() {
        bail!(
            "{program} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    Ok(text)
}
