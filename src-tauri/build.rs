fn main() {
    build_mask_helper();
    tauri_build::build()
}

/// On macOS, masking uses the Vision framework through a small Swift executable
/// (`helpers/mask_macos.swift`) bundled as a sidecar. It must exist before
/// `tauri_build` runs, since that validates the `externalBin` entries.
fn build_mask_helper() {
    println!("cargo:rerun-if-changed=helpers/mask_macos.swift");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let target = std::env::var("TARGET").unwrap();
    let swift_target = match target.as_str() {
        "aarch64-apple-darwin" => "arm64-apple-macos14.0",
        "x86_64-apple-darwin" => "x86_64-apple-macos14.0",
        other => panic!("unsupported macOS target: {other}"),
    };
    let out = format!("binaries/splatforge_mask-{target}");
    std::fs::create_dir_all("binaries").unwrap();
    let status = std::process::Command::new("xcrun")
        .args(["swiftc", "-O", "-target", swift_target, "helpers/mask_macos.swift", "-o", &out])
        .status()
        .expect("swiftc not found; install the Xcode Command Line Tools");
    assert!(status.success(), "failed to compile helpers/mask_macos.swift");
}
