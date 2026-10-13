# Retouch with ComfyUI

*Web version only, for now.*

After training, the **Retouch** panel improves angles the orbit video didn't
cover well (from below, from above, or the view you aimed at in 3D). The flow:

1. **Generate a camera path** that starts at a real photo, goes through those
   angles and comes back to a real photo. The motion is picked from a select:
   - **From this view:** goes through the current 3D camera. The **Frame**
     button in the 3D view darkens what falls outside the video frame, to help
     aiming.
   - **360° turn:** starts at the current view and circles the subject at the
     same height until it comes back. With the distance set to "Current view's", it is good for refining up close: the first frame is exactly what
     Frame shows.
   - **Tilted 360° turn:** at the current view's tilt (from above or below), in
     three phases. It rises straight from the photos' height to the view on an
     arc around the point the view aims at (like a crane), with the camera
     starting level and gradually tilting to the view's angle. Then it loops
     around that point at that tilt, and finally comes back down. The first
     frame (the video model's reference) sits at a height well covered by the
     photos, already framing what the view frames.
   - **Arc below / Arc above:** a third of the orbit around the nearest photo,
     deviating in height.
   - **360° turn below / above / up and down:** the full loop at another
     height, starting and ending facing the same photo.
   - **Record motion:** **Record** samples the 3D camera every frame while you
     move it (drag, wheel, Spin) until **Stop** (or 30 s). The still start and
     the glide after the last input are cut. The recording is then resampled
     to the clip, with these options:
     - **Uniform speed:** 0% keeps the recorded pace, 100% moves at a constant
       speed along the path with no pauses. Speed counts zooms too, as how much
       the image changes.
     - **Stabilization:** smooths position and aim (at 100%, a Gaussian
       with sigma 10% of the recording); the first and last frames stay where
       they were recorded.
     - **Only while the button is pressed:** the clock runs only with a button
       pressed on the 3D view, or while the camera still glides after you
       release it (wheel zoom glides too), so pauses between strokes don't
       count.
     - **Fixed duration:** keeps the Duration field and fits the motion to it,
       instead of using the recorded time.

   The **Anchor** button in the 3D view shows the point the camera orbits
   around with a move gizmo. Drag its axes to place it (on the head, say): on
   release the camera turns to aim at it. Once placed, the preset motions turn
   around it too. The 360° turns circle its vertical axis. Arcs and turns at
   another height still start and end at real photos and aim at the anchor in
   between. **Recenter** clears it. **Lock angle** starts off.

   Duration (in seconds, rounded to 8n + 1 frames), tilt and distance sit next
   to it, and the path is rebuilt when they change. Except in recorded motions,
   distance holds for the whole clip, with no zooming in or out: either the photos' distance or the current
   view's (always the view's for "From this view"). The camera doesn't stop at
   the ends and moves at a constant angular speed even when the photos are
   unevenly spaced. The panel shows the peak speed and warns above 30°/s (the
   video model blurs fast motion), suggesting a duration that fixes it.
2. **Render the splat along the path:** this is the reference video.
   - Rendering uses Spark off-screen, blending in sRGB like the screen and like
     Brush.
   - The video is encoded as MP4 (Mediabunny) at the fps chosen in the panel
     (default: the workflow's). Aspect ratio (the lens', or 9:16 to 16:9,
     keeping the photos' frame height) and resolution (long side from 512 to
     1920, multiples of 32) are chosen in the panel; the default is the lens at
     480×960.
3. **Send the video to ComfyUI** with a restoration workflow.
   - Set it up in **Integrations**: the ComfyUI address and the `.json` exported
     with *Export (API)*, plus each workflow's input and output nodes.
   - **Custom fields** (optional, per workflow): pick a node and field (prompt,
     seed, LoRA strength...) and give it a name. They show up in the retouch
     panel for editing before sending; the workflow's value is the default, and
     changes are saved in this browser.
   - The workflow should follow the input video (duration, resolution and
     aspect ratio); the app only swaps the file in the input node. The reference
     video's default fps is the workflow's first fixed `fps` (30 in the LTX
     one), or 24.
   - The restored video must have the same number of frames and aspect ratio as
     the reference. Its size may change: fine-tuning uses the frames at the size
     they arrive in, with intrinsics scaled to match.
   - Progress comes through ComfyUI's WebSocket.
4. **Apply retouch** fine-tunes the splat from itself, using the photos plus the
   restored frames.
   - Uses Brush with `init.ply`, 3k steps and a 10× smaller `lr-mean`, in a
     separate `dataset_retouch/`.
   - Only some of the restored frames are used: by default up to 25% of the
     number of photos, split across clips and evenly spaced (excluding the
     ends). Too many restored frames pull the splat towards details the video
     model invented, and the centre, which the photos see well, loses
     sharpness. In tests, using 1 in every 3 frames recovered 2 dB on the photos
     and kept the ghost removed. The limit is adjustable in the panel.
   - The previous splat is kept: **Undo retouch** goes back to it and **Compare
     with before** shows both.

## Example workflows

`assets/workflows/` has two ComfyUI workflows to start from. Model files are
not included; download them into your ComfyUI `models/` folders.

| File | Format | What it does |
|---|---|---|
| `ltx-2.5-refinement.json` | API (*Export (API)*) | Restoration workflow for **Retouch**: LTX 2.5 with the IC-LoRA decompression guide re-renders the reference video. Import it in **Integrations**; the input node is the `LoadVideo` and the output is the `SaveVideo`. |
| `minimax-h3-360-orbit.json` | UI (open in ComfyUI) | Generates an orbit video from a single photo with MiniMax-H3 and a 360° orbit LoRA. The result can be dropped into Gausify Orbit as the input video. |

ComfyUI must run with `--enable-cors-header` so the browser can call it. In
Chrome, the HTTPS page also asks for local network permission to reach
`http://127.0.0.1`. To develop without a GPU there is a mock ComfyUI:
`node scripts/comfy-mock.mjs [--output video.mp4]`, or the `comfy-mock` entry
in `.claude/launch.json`.

`scripts/web-bench/retouch.mjs` runs the whole flow in headless Chrome against
`pnpm web:dev` with `WEB_FS_ANY=1`. It opens a `.gorbit`, creates the clip,
restores, applies and saves the splat. The desktop app's panel can't be used
for this, since WebGPU training doesn't run there.
