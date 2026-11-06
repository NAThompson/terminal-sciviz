/** Minimal PNG: RGBA8 encoder and 8-bit grey/GA/RGB/RGBA decoder (zlib via Compression Streams). */

const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k += 1) {
    c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  return c >>> 0;
});

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (const b of bytes) {
    c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

async function pipe(data: Uint8Array, stream: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const out = new Response(new Blob([data as BlobPart]).stream().pipeThrough(stream));
  return new Uint8Array(await out.arrayBuffer());
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const v = new DataView(out.buffer);
  v.setUint32(0, data.length);
  out.set(new TextEncoder().encode(type), 4);
  out.set(data, 8);
  v.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** PNG of straight-alpha RGBA8 pixels, top row first (filter "up" on every row). */
export async function encodePng(rgba: Uint8Array, width: number, height: number): Promise<Uint8Array> {
  const stride = width * 4;
  const raw = new Uint8Array((stride + 1) * height);
  for (let y = 0; y < height; y += 1) {
    const o = y * (stride + 1);
    raw[o] = 2; // up
    const row = y * stride;
    if (y === 0) {
      raw.set(rgba.subarray(0, stride), o + 1);
    } else {
      for (let i = 0; i < stride; i += 1) {
        raw[o + 1 + i] = rgba[row + i] - rgba[row - stride + i];
      }
    }
  }
  const ihdr = new Uint8Array(13);
  const v = new DataView(ihdr.buffer);
  v.setUint32(0, width);
  v.setUint32(4, height);
  ihdr.set([8, 6, 0, 0, 0], 8); // 8 bits, RGBA
  const idat = await pipe(raw, new CompressionStream("deflate")); // zlib stream
  const parts = [
    Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10),
    chunk("IHDR", ihdr),
    chunk("IDAT", idat),
    chunk("IEND", new Uint8Array(0)),
  ];
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

/** Decodes a non-interlaced 8-bit PNG to RGBA8. */
export async function decodePng(file: Uint8Array): Promise<{ width: number; height: number; rgba: Uint8Array }> {
  const v = new DataView(file.buffer, file.byteOffset, file.byteLength);
  let o = 8;
  let width = 0;
  let height = 0;
  let channels = 0;
  const idat: Uint8Array[] = [];
  while (o < file.length) {
    const len = v.getUint32(o);
    const type = new TextDecoder().decode(file.subarray(o + 4, o + 8));
    const data = file.subarray(o + 8, o + 8 + len);
    if (type === "IHDR") {
      width = v.getUint32(o + 8);
      height = v.getUint32(o + 12);
      const [depth, color, , , interlace] = data.subarray(8);
      channels = { 0: 1, 2: 3, 4: 2, 6: 4 }[color as 0 | 2 | 4 | 6] ?? 0;
      if (depth !== 8 || !channels || interlace) {
        throw new Error("Only non-interlaced 8-bit grey/RGB(A) PNGs");
      }
    } else if (type === "IDAT") {
      idat.push(data);
    }
    o += 12 + len;
  }
  const joined = new Uint8Array(idat.reduce((n, d) => n + d.length, 0));
  idat.reduce((at, d) => (joined.set(d, at), at + d.length), 0);
  const raw = await pipe(joined, new DecompressionStream("deflate"));

  const bpp = channels;
  const stride = width * bpp;
  const px = new Uint8Array(stride * height);
  for (let y = 0; y < height; y += 1) {
    const filter = raw[y * (stride + 1)];
    const src = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const row = y * stride;
    for (let i = 0; i < stride; i += 1) {
      const a = i >= bpp ? px[row + i - bpp] : 0;
      const b = y > 0 ? px[row - stride + i] : 0;
      const c = i >= bpp && y > 0 ? px[row - stride + i - bpp] : 0;
      let pred = 0;
      if (filter === 1) pred = a;
      else if (filter === 2) pred = b;
      else if (filter === 3) pred = (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      px[row + i] = (src[i] + pred) & 0xff;
    }
  }
  const rgba = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i += 1) {
    const s = px.subarray(i * bpp, i * bpp + bpp);
    if (bpp === 1) rgba.set([s[0], s[0], s[0], 255], i * 4);
    else if (bpp === 2) rgba.set([s[0], s[0], s[0], s[1]], i * 4);
    else if (bpp === 3) rgba.set([s[0], s[1], s[2], 255], i * 4);
    else rgba.set(s, i * 4);
  }
  return { width, height, rgba };
}
