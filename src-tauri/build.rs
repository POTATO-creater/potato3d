fn main() {
    build_mask_helper();
    tauri_build::build()
}

/// No macOS o recorte usa o framework Vision via um pequeno executável Swift
/// (`helpers/mask_macos.swift`), empacotado como sidecar. Ele precisa existir
/// antes do `tauri_build`, que valida os `externalBin`.
fn build_mask_helper() {
    println!("cargo:rerun-if-changed=helpers/mask_macos.swift");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let target = std::env::var("TARGET").unwrap();
    let swift_target = match target.as_str() {
        "aarch64-apple-darwin" => "arm64-apple-macos14.0",
        "x86_64-apple-darwin" => "x86_64-apple-macos14.0",
        other => panic!("alvo macOS não suportado: {other}"),
    };
    let out = format!("binaries/splatforge_mask-{target}");
    std::fs::create_dir_all("binaries").unwrap();
    let status = std::process::Command::new("xcrun")
        .args(["swiftc", "-O", "-target", swift_target, "helpers/mask_macos.swift", "-o", &out])
        .status()
        .expect("swiftc não encontrado; instale as Command Line Tools do Xcode");
    assert!(status.success(), "falha ao compilar helpers/mask_macos.swift");
}
