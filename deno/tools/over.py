# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow"]
# ///
"""Composite straight-alpha PNGs over #101216 (the terminal background), for viewing.

    uv run deno/tools/over.py IN.png [IN2.png ...]   # writes IN.over.png next to each
"""
import sys
from pathlib import Path

from PIL import Image

for p in map(Path, sys.argv[1:]):
    img = Image.open(p).convert("RGBA")
    bg = Image.new("RGBA", img.size, (0x10, 0x12, 0x16, 255))
    out = p.with_suffix(".over.png")
    Image.alpha_composite(bg, img).convert("RGB").save(out)
    print(out)
