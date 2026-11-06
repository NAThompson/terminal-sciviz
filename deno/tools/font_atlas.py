# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow", "fonttools", "scipy", "numpy"]
# ///
"""Signed-distance-field font atlas for lib/overlay.ts (no canvas in Deno: text is drawn on the GPU).

    uv run deno/tools/font_atlas.py      # -> deno/assets/font.png + deno/assets/font.json

Fonts (SIL Open Font License 1.1; licences and provenance: LICENSES/README.md and
deno/assets/fonts/*/OFL.txt): JetBrains Mono Regular and Bold, and for what it lacks (math
italics, ℜ ℑ, some scripts) STIX Two Math, on JetBrains Mono's baseline. Only those files are
read, never system fonts. One size per face: each glyph rendered by FreeType at OVERSAMPLE × BASE
px, thresholded, its signed Euclidean distance transform (scipy) taken there and averaged down to
BASE px, plus WEIGHT[face] px (strokes as heavy as the Menlo atlas this replaced). font.png is
8-bit grey: 128 + 127 · d / SPREAD, d the distance in BASE px to the outline (positive inside), so
an edge at any scale is `v = 0.5`. font.json has, per face, the base size, ascent/descent (BASE
px: LINE_BOX, the line box layouts were made for) and per glyph its atlas rectangle (with SPREAD px
of field around the ink), its offset from the pen on the baseline and its (float) advance.

Scripts: sub/superscripts neither font has (ₓ ₐ ₑ ₒ ᵢ ⱼ ᵏ ⁿ) are drawn from the regular glyph
(`x` for `ₓ`) at the font's script size, on the baseline of its `₀` or `⁰` (signs centred on
the digits). Small sizes: scripts
(about 0.46 em tall) fill in below ~16 px, "₀" reading as "₈". Each face also gets `small`
variants of every script, drawn the same way at SCRIPT_GAIN × the script size and emboldened by
SCRIPT_BOLD (BASE px): larger, with open counters. lib/overlay.ts uses them below SMALL_PX px per
em. Characters no font covers make the run fail.
"""


import json
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

ASSETS = Path(__file__).resolve().parents[1] / "assets"
FONTS = ASSETS / "fonts"
FACES = {
    "regular": FONTS / "JetBrainsMono" / "JetBrainsMono-Regular.ttf",
    "bold": FONTS / "JetBrainsMono" / "JetBrainsMono-Bold.ttf",
}
FALLBACK = FONTS / "STIXTwoMath" / "STIXTwoMath-Regular.otf"
BASE = 48  # px per em of the atlas
SPREAD = 8  # px (at BASE) of distance encoded either side of the outline
OVERSAMPLE = 8  # distance transform taken at BASE × OVERSAMPLE
WEIGHT = {"regular": 0.2, "bold": 0.45}  # BASE px added to the field: stems as Menlo's
LINE_BOX = (0.929, 0.237)  # ascent, descent (em): Menlo's, which the layouts were tuned to
CHARS = "".join(chr(c) for c in range(32, 127)) + "·−×Δ≈…±µ–—"
# Math: the face where it has them, else STIX Two Math (same pixel size and baseline)
CHARS += "π≤≥∞θτωφψκλμσ→←√½∂∫ℜℑ‖∇′″𝜑𝜋𝜏𝜔𝜅𝜃𝜓𝑥𝑦𝑧𝑡𝑘𝑓𝑖𝑛𝑠𝑢𝐽𝐸𝐻𝐶𝑆𝐫𝑝𝑞𝑚𝑤𝑣𝑎𝑏𝑐𝑑𝑒𝑔𝑟𝑗𝐦𝐯𝐤"
# Greek, upright and math italic (titles such as "Lorenz attractor (𝜎 = 10, 𝜌 = 28, 𝛽 = 8/3)")
CHARS += "αβγδεζηικνξορςυχΓΘΛΞΠΣΦΨΩ𝛼𝛽𝛾𝛿𝜀𝜁𝜂𝜄𝜆𝜇𝜈𝜉𝜌𝜎𝜒Ω"
# Script glyph -> the regular glyph its synthesized and small forms are drawn from, and sub/sup
SCRIPTS = {
    **{s: (d, "sub") for s, d in zip("₀₁₂₃₄₅₆₇₈₉₊₋ₓₐₑₒᵢⱼ", "0123456789+-xaeoij")},
    **{s: (d, "sup") for s, d in zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁱᵏⁿ", "0123456789+-ikn")},
}
SCRIPT_GAIN = 1.2  # × the script size
SCRIPT_BOLD = 0.5  # BASE px added to the field (strokes 1 px thicker at BASE)
SMALL_PX = 16  # px per em below which the small script variants are drawn
WIDTH = 2048
PAD = 1


def sdf(font, ch, bold=0.0):
    """Field of `ch` at BASE px (uint8), its offset from the pen (BASE px, whole), or None."""
    k = OVERSAMPLE
    l, t, r, b = font.getbbox(ch, anchor="ls")  # at BASE × k px
    if r <= l or b <= t:
        return None
    # Ink box in BASE px (whole), plus the spread
    L, T = (l // k) - SPREAD, (t // k) - SPREAD
    R, B = -(-r // k) + SPREAD, -(-b // k) + SPREAD
    img = Image.new("L", ((R - L) * k, (B - T) * k), 0)
    ImageDraw.Draw(img).text((-L * k, -T * k), ch, font=font, fill=255, anchor="ls")
    ink = np.asarray(img) >= 128
    # Signed distance (hi-res px) from each pixel centre to the outline, positive inside
    inside = distance_transform_edt(ink) - 0.5
    outside = distance_transform_edt(~ink) - 0.5
    d = np.where(ink, inside, -outside) / k  # BASE px
    d = d.reshape(B - T, k, R - L, k).mean(axis=(1, 3)) + bold
    v = np.clip(np.round(128 + 127 * d / SPREAD), 0, 255).astype(np.uint8)
    return Image.fromarray(v, "L"), L, T


def ink_box(font, ch):
    """Ink bounds (left, top, right, bottom) of `ch` from the pen on the baseline, font px."""
    l, t, r, b = font.getbbox(ch, anchor="ls")
    img = Image.new("L", (r - l + 4, b - t + 4), 0)
    ImageDraw.Draw(img).text((2 - l, 2 - t), ch, font=font, fill=255, anchor="ls")
    ys, xs = np.nonzero(np.asarray(img) >= 128)
    return xs.min() - 2 + l, ys.min() - 2 + t, xs.max() + 1 - 2 + l, ys.max() + 1 - 2 + t


def script_sdf(path, base, kind, gain, bold):
    """`base` at `gain` × the face's script size, its digits on the baseline of the face's `₀`
    (subscripts) or `⁰` (superscripts), centred in the advance."""
    face = ImageFont.truetype(str(path), BASE * OVERSAMPLE)
    _, zt, _, zb = ink_box(face, "₀" if kind == "sub" else "⁰")
    _, ot, _, ob = ink_box(face, "0")
    ratio = (zb - zt) / (ob - ot)  # size of the face's scripts, about 0.63
    scaled = ImageFont.truetype(str(path), round(BASE * OVERSAMPLE * ratio * gain))
    _, st, _, sb = ink_box(scaled, "0")
    field, L, T = sdf(scaled, base, bold)
    dx = (face.getlength("0") - scaled.getlength(base)) / 2 / OVERSAMPLE
    dy = (zb - sb) / OVERSAMPLE
    if base in "+-":  # signs centred on the digits (as in 10⁻³), not on the x-height
        _, bt, _, bb = ink_box(scaled, base)
        dy += ((st + sb) - (bt + bb)) / 2 / OVERSAMPLE
    return field, L + round(dx), T + round(dy)


def main():
    glyphs = []  # (face, key, image, bx, by, advance), key ("small", ch) for small variants
    metrics = {}
    stix = TTFont(FALLBACK).getBestCmap()
    fallback = ImageFont.truetype(str(FALLBACK), BASE * OVERSAMPLE)
    for face, path in FACES.items():
        cmap = TTFont(path).getBestCmap()
        primary = ImageFont.truetype(str(path), BASE * OVERSAMPLE)
        metrics[face] = {"size": BASE, "ascent": LINE_BOX[0] * BASE, "descent": LINE_BOX[1] * BASE, "glyphs": {}}
        w = WEIGHT[face]
        missing = []
        for ch in dict.fromkeys(CHARS + "".join(SCRIPTS)):
            adv = primary.getlength("0") / OVERSAMPLE  # monospaced
            if ord(ch) in cmap:
                field = sdf(primary, ch, w)
                adv = primary.getlength(ch) / OVERSAMPLE
            elif ch in SCRIPTS:  # drawn from the base glyph, at the script size
                field = script_sdf(path, *SCRIPTS[ch], 1.0, w)
            elif ord(ch) in stix:
                field = sdf(fallback, ch, w)
                adv = fallback.getlength(ch) / OVERSAMPLE
            else:
                missing.append(ch)
                continue
            glyphs.append((face, ch, *(field or (None, 0, 0)), adv))
        if missing:
            raise SystemExit(f"{face}: no font has {''.join(missing)!r}")
        for script, (base, kind) in SCRIPTS.items():
            adv = primary.getlength("0") / OVERSAMPLE
            glyphs.append((face, ("small", script), *script_sdf(path, base, kind, SCRIPT_GAIN, w + SCRIPT_BOLD), adv))

    # Shelf packing, tallest first
    order = sorted(range(len(glyphs)), key=lambda i: -(glyphs[i][2].height if glyphs[i][2] else 0))
    x = y = shelf = 0
    places = {}
    for i in order:
        img = glyphs[i][2]
        if img is None:
            continue
        if x + img.width + PAD > WIDTH:
            x, y, shelf = 0, y + shelf + PAD, 0
        places[i] = (x, y)
        x += img.width + PAD
        shelf = max(shelf, img.height)
    height = y + shelf
    atlas = Image.new("L", (WIDTH, height), 0)
    for i, (face, ch, img, bx, by, adv) in enumerate(glyphs):
        entry = {"advance": round(adv, 4)}
        if img is not None:
            px, py = places[i]
            atlas.paste(img, (px, py))
            entry.update(x=px, y=py, w=img.width, h=img.height, bx=bx, by=by)
        if isinstance(ch, tuple):
            metrics[face].setdefault("small", {})[ch[1]] = entry
        else:
            metrics[face]["glyphs"][ch] = entry

    atlas.save(ASSETS / "font.png", optimize=True)
    (ASSETS / "font.json").write_text(json.dumps(
        {"font": "JetBrains Mono, STIX Two Math", "sdf": {"spread": SPREAD}, "smallPx": SMALL_PX, "width": WIDTH, "height": height,
         "faces": metrics},
        ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {ASSETS / 'font.png'} ({WIDTH}x{height}), {len(glyphs)} glyphs")


if __name__ == "__main__":
    main()
