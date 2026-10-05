// Recorte do sujeito no macOS com o framework Vision (o mesmo do "copiar
// sujeito" do Fotos). Compilado pelo build.rs como sidecar do Tauri.
//
//   splatforge_mask <pasta_entrada> <pasta_saida>
//
// Para cada .jpg/.png da entrada grava <nome>.png em tons de cinza
// (255 = sujeito). Imprime "progress i/n" e "nosubject <arquivo>" em stdout.
import CoreImage
import CoreML
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

let args = CommandLine.arguments
guard args.count == 3 else { fail("uso: splatforge_mask <entrada> <saida>") }
let inDir = URL(fileURLWithPath: args[1])
let outDir = URL(fileURLWithPath: args[2])
try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

let files = ((try? FileManager.default.contentsOfDirectory(atPath: inDir.path)) ?? [])
    .filter { $0.lowercased().hasSuffix(".jpg") || $0.lowercased().hasSuffix(".png") }
    .sorted()
if files.isEmpty { fail("nenhuma imagem em \(inDir.path)") }

let request = VNGenerateForegroundInstanceMaskRequest()
// Roda na GPU: previsível e não depende da fila do compilador do Neural Engine.
if ProcessInfo.processInfo.environment["SPLATFORGE_MASK_DEVICE"] != "auto",
   let gpu = MLComputeDevice.allComputeDevices.first(where: { if case .gpu = $0 { return true }; return false }),
   let stages = try? request.supportedComputeStageDevices {
    for stage in stages.keys { request.setComputeDevice(gpu, for: stage) }
}

let context = CIContext(options: [.workingColorSpace: NSNull()])
let gray = CGColorSpaceCreateDeviceGray()

for (i, name) in files.enumerated() {
    autoreleasepool {
        let url = inDir.appendingPathComponent(name)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else { fail("não foi possível ler \(name)") }
        let handler = VNImageRequestHandler(cgImage: image)
        let out = outDir.appendingPathComponent((name as NSString).deletingPathExtension + ".png")
        var mask: CGImage?
        do {
            try handler.perform([request])
            if let obs = request.results?.first, !obs.allInstances.isEmpty {
                let buf = try obs.generateScaledMaskForImage(forInstances: obs.allInstances, from: handler)
                let ci = CIImage(cvPixelBuffer: buf)
                mask = context.createCGImage(ci, from: ci.extent, format: .L8, colorSpace: gray)
            }
        } catch {
            fail("Vision falhou em \(name): \(error)")
        }
        if mask == nil {
            // Sem sujeito detectado: mantém o frame inteiro (máscara branca).
            print("nosubject \(name)")
            let ctx = CGContext(data: nil, width: image.width, height: image.height, bitsPerComponent: 8,
                                bytesPerRow: 0, space: gray, bitmapInfo: CGImageAlphaInfo.none.rawValue)!
            ctx.setFillColor(gray: 1, alpha: 1)
            ctx.fill(CGRect(x: 0, y: 0, width: image.width, height: image.height))
            mask = ctx.makeImage()
        }
        guard let dest = CGImageDestinationCreateWithURL(out as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { fail("não foi possível gravar \(out.path)") }
        CGImageDestinationAddImage(dest, mask!, nil)
        if !CGImageDestinationFinalize(dest) { fail("não foi possível gravar \(out.path)") }
        print("progress \(i + 1)/\(files.count)")
        fflush(stdout)
    }
}
