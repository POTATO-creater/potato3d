#!/usr/bin/env python3
"""Generates the favicon and PWA icons (web/public/) from assets/icon.png.

Usage: python3 scripts/web-icons.py

- favicon.ico, icon-192.png, icon-512.png: the icon's rounded tile, without the
  transparent margin of the macOS template (reads better at small sizes).
- icon-maskable-*.png and apple-touch-icon.png: opaque and without corners,
  because the OS applies its own mask. The sphere stays inside the safe zone (80%).
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageStat

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "icon.png"
OUT = ROOT / "web" / "public"


def tile(src: Image.Image) -> Image.Image:
    """Crops the rounded tile (everything that is nearly opaque)."""
    box = src.getchannel("A").point(lambda a: 255 if a > 200 else 0).getbbox()
    return src.crop(box)


def full_bleed(t: Image.Image) -> Image.Image:
    """Tile without the rounded corners: the background is interpolated from a 3×3
    grid of colors sampled inside the tile itself (following its vignette)."""
    w, h = t.size
    edge, corner, r = w // 12, w // 5, w // 40
    grid = Image.new("RGB", (3, 3))
    for gy in range(3):
        for gx in range(3):
            # At the corners, sample further in, away from the transparent curve;
            # the tile's center is the sphere, so use the top edge instead.
            inset = corner if gx != 1 and gy != 1 else edge
            x = (inset, w // 2, w - inset)[gx]
            y = edge if (gx, gy) == (1, 1) else (inset, h // 2, h - inset)[gy]
            box = (x - r, y - r, x + r, y + r)
            mean = ImageStat.Stat(t.crop(box).convert("RGB"), t.getchannel("A").crop(box)).mean
            grid.putpixel((gx, gy), tuple(round(v) for v in mean))
    bg = grid.resize((w, h), Image.BICUBIC).convert("RGBA")
    # Only the sphere and its surroundings come from the tile, in a feathered
    # disc: the tile's edge never shows.
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((w * 0.08, h * 0.08, w * 0.92, h * 0.92), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(w * 0.04))
    return Image.composite(t, bg, mask).convert("RGB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = tile(Image.open(SRC).convert("RGBA"))
    solid = full_bleed(t)
    size = lambda im, s: im.resize((s, s), Image.LANCZOS)

    for s in (192, 512):
        size(t, s).save(OUT / f"icon-{s}.png", optimize=True)
        size(solid, s).save(OUT / f"icon-maskable-{s}.png", optimize=True)
    size(solid, 180).save(OUT / "apple-touch-icon.png", optimize=True)
    size(t, 256).save(OUT / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    for f in sorted(OUT.glob("*.png")) + [OUT / "favicon.ico"]:
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1024:.1f} kB")


if __name__ == "__main__":
    main()
