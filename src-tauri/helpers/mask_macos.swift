// Subject masking on macOS with the Vision framework (the same one behind
// Photos' "Copy Subject"). Built by build.rs as a Tauri sidecar.
//
//   splatforge_mask <input_dir> <output_dir>
//
// For each input .jpg/.png, writes a grayscale <name>.png (255 = subject).
// Prints "progress i/n" and "nosubject <file>" to stdout.
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
guard args.count == 3 else { fail("usage: splatforge_mask <input> <output>") }
let inDir = URL(fileURLWithPath: args[1])
let outDir = URL(fileURLWithPath: args[2])
try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

let files = ((try? FileManager.default.contentsOfDirectory(atPath: inDir.path)) ?? [])
    .filter { $0.lowercased().hasSuffix(".jpg") || $0.lowercased().hasSuffix(".png") }
    .sorted()
if files.isEmpty { fail("no images in \(inDir.path)") }

let request = VNGenerateForegroundInstanceMaskRequest()
// Run on the GPU: predictable, and doesn't depend on the Neural Engine compiler queue.
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
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else { fail("could not read \(name)") }
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
            fail("Vision failed on \(name): \(error)")
        }
        if mask == nil {
            // No subject detected: keep the whole frame (white mask).
            print("nosubject \(name)")
            let ctx = CGContext(data: nil, width: image.width, height: image.height, bitsPerComponent: 8,
                                bytesPerRow: 0, space: gray, bitmapInfo: CGImageAlphaInfo.none.rawValue)!
            ctx.setFillColor(gray: 1, alpha: 1)
            ctx.fill(CGRect(x: 0, y: 0, width: image.width, height: image.height))
            mask = ctx.makeImage()
        }
        guard let dest = CGImageDestinationCreateWithURL(out as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { fail("could not write \(out.path)") }
        CGImageDestinationAddImage(dest, mask!, nil)
        if !CGImageDestinationFinalize(dest) { fail("could not write \(out.path)") }
        print("progress \(i + 1)/\(files.count)")
        fflush(stdout)
    }
}
