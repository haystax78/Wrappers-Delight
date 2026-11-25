"""Minimal Photoshop (PSD / PSB) composite reader and writer (numpy).

Blender reads PSD files but cannot write them, and how its loader treats the
merged image's alpha depends on the OpenImageIO version.  Texture Bake reads
and writes the file's own merged (composite) image here instead, so raw
values go in and come out unchanged.

Supported: Grayscale and RGB, 8/16/32 bits per channel, composite compression
raw, RLE (PackBits), ZIP and ZIP with prediction.  Written files are flat
(merged image only, raw), same version (PSD/PSB), mode and depth, keeping the
resolution and ICC profile resources.  Only the first extra channel (alpha)
is kept; further alpha channels / spot channels are dropped.
"""

import struct
import zlib

import numpy as np

MODE_GRAY = 1
MODE_RGB = 3
_MODE_NAMES = {0: "Bitmap", 1: "Grayscale", 2: "Indexed", 3: "RGB",
               4: "CMYK", 7: "Multichannel", 8: "Duotone", 9: "Lab"}

# Image resources copied to the output: resolution, ICC profile, ICC
# untagged flag; alpha channel names / ids only when every channel is kept.
_KEEP = {1005, 1039, 1041}
_KEEP_ALPHA = {1006, 1045, 1053}


class PsdError(Exception):
    pass


def is_psd(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"8BPS"
    except OSError:
        return False


class PsdInfo:
    """Header fields plus the resource blocks worth carrying over."""

    def __init__(self, version, channels, height, width, depth, mode,
                 resources):
        self.version = version
        self.channels = channels
        self.height = height
        self.width = width
        self.depth = depth
        self.mode = mode
        self.resources = resources      # [(id, raw block bytes)]

    @property
    def base(self):
        return 1 if self.mode == MODE_GRAY else 3

    @property
    def has_alpha(self):
        return self.channels > self.base

    @property
    def kept(self):
        """Channels carried through the bake (colour + first alpha)."""
        return self.base + (1 if self.has_alpha else 0)


def _resource_blocks(data):
    """Split the image resources section into (id, raw bytes) blocks."""
    out, i, n = [], 0, len(data)
    while i + 12 <= n and data[i:i + 4] in (b"8BIM", b"MeSa", b"AgHg",
                                            b"PHUT", b"DCSR"):
        start = i
        rid = struct.unpack_from(">H", data, i + 4)[0]
        name_len = data[i + 6]
        j = i + 6 + ((name_len + 2) & ~1)          # pascal string, even
        size = struct.unpack_from(">I", data, j)[0]
        end = j + 4 + size + (size & 1)
        out.append((rid, bytes(data[start:end])))
        i = end
    return out


def _unpackbits(src, size):
    """PackBits decode of a whole stream (rows are encoded independently,
    so decoding their concatenation gives the concatenated rows)."""
    out = bytearray()
    i, n = 0, len(src)
    while i < n:
        h = src[i]
        i += 1
        if h < 128:
            out += src[i:i + h + 1]
            i += h + 1
        elif h > 128:
            out += src[i:i + 1] * (257 - h)
            i += 1
    if len(out) < size:
        out += bytes(size - len(out))
    return bytes(out[:size])


def _unpredict(raw, info, nplanes):
    """Undo ZIP-with-prediction row deltas for ``nplanes`` planes."""
    w, h, d = info.width, info.height, info.depth
    if d == 8:
        a = np.frombuffer(raw, np.uint8, nplanes * h * w).reshape(-1, w)
        return np.cumsum(a, axis=1, dtype=np.uint8).tobytes()
    if d == 16:
        a = np.frombuffer(raw, ">u2", nplanes * h * w).reshape(-1, w)
        a = np.cumsum(a.astype(np.uint16), axis=1, dtype=np.uint16)
        return a.astype(">u2").tobytes()
    # 32-bit: bytes delta-coded per row, stored as 4 byte planes per row.
    a = np.frombuffer(raw, np.uint8, nplanes * h * w * 4).reshape(-1, w * 4)
    a = np.cumsum(a, axis=1, dtype=np.uint8)
    a = a.reshape(-1, 4, w).transpose(0, 2, 1)
    return np.ascontiguousarray(a).tobytes()


def read(path):
    """Return (PsdInfo, float32 array (kept, height, width)) of the merged
    image, top row first, values 0..1 for 8/16-bit, raw floats for 32-bit."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 26 or data[:4] != b"8BPS":
        raise PsdError("not a Photoshop file")
    version, = struct.unpack_from(">H", data, 4)
    if version not in (1, 2):
        raise PsdError(f"unknown PSD version {version}")
    channels, height, width, depth, mode = struct.unpack_from(">HIIHH",
                                                              data, 12)
    if mode not in (MODE_GRAY, MODE_RGB):
        raise PsdError(f"{_MODE_NAMES.get(mode, mode)} mode is not "
                       "supported (convert to RGB or Grayscale)")
    if depth not in (8, 16, 32):
        raise PsdError(f"{depth}-bit is not supported")
    i = 26
    cm_len, = struct.unpack_from(">I", data, i)
    i += 4 + cm_len
    res_len, = struct.unpack_from(">I", data, i)
    blocks = _resource_blocks(memoryview(data)[i + 4:i + 4 + res_len])
    i += 4 + res_len
    if version == 1:
        lm_len, = struct.unpack_from(">I", data, i)
        i += 4 + lm_len
    else:
        lm_len, = struct.unpack_from(">Q", data, i)
        i += 8 + lm_len
    comp, = struct.unpack_from(">H", data, i)
    i += 2

    info = PsdInfo(version, channels, height, width, depth, mode, blocks)
    if channels < info.base:
        raise PsdError("missing colour channels")
    k = info.kept
    bpc = depth // 8
    plane = width * height * bpc

    if comp == 0:
        raw = data[i:i + k * plane]
    elif comp == 1:
        cnt_size = 2 if version == 1 else 4
        counts = np.frombuffer(data, ">u2" if cnt_size == 2 else ">u4",
                               channels * height, i).astype(np.int64)
        start = i + channels * height * cnt_size
        end = start + int(counts[:k * height].sum())
        raw = _unpackbits(data[start:end], k * plane)
    elif comp in (2, 3):
        raw = zlib.decompress(data[i:])
        if comp == 3:
            raw = _unpredict(raw, info, k)
    else:
        raise PsdError(f"unknown compression {comp}")
    if len(raw) < k * plane:
        raise PsdError("image data is truncated")

    if depth == 8:
        px = np.frombuffer(raw, np.uint8, k * width * height)
        px = px.astype(np.float32) / 255.0
    elif depth == 16:
        px = np.frombuffer(raw, ">u2", k * width * height)
        px = px.astype(np.float32) / 65535.0
    else:
        px = np.frombuffer(raw, ">f4", k * width * height).astype(np.float32)
    return info, px.reshape(k, height, width)


def write(path, info, planes):
    """Write a flat PSD/PSB: ``planes`` = float array (info.kept, h, w), top
    row first, in the source's value range."""
    k, h, w = planes.shape
    resources = b"".join(
        raw for rid, raw in info.resources
        if rid in _KEEP or (rid in _KEEP_ALPHA and info.channels == k))
    if info.depth == 8:
        body = np.rint(np.clip(planes, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif info.depth == 16:
        body = np.rint(np.clip(planes, 0.0, 1.0) * 65535.0).astype(">u2")
    else:
        body = planes.astype(">f4")
    with open(path, "wb") as f:
        f.write(b"8BPS" + struct.pack(">H", info.version) + bytes(6))
        f.write(struct.pack(">HIIHH", k, h, w, info.depth, info.mode))
        f.write(struct.pack(">I", 0))                   # colour mode data
        f.write(struct.pack(">I", len(resources)) + resources)
        f.write(struct.pack(">I" if info.version == 1 else ">Q", 0))
        f.write(struct.pack(">H", 0))                   # raw
        f.write(body.tobytes())
