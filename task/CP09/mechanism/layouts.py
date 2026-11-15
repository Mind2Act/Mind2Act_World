"""Deterministic episode geometry. Target geometry stays evaluator-private."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw

CASE_ROOT = Path(__file__).resolve().parents[1]

CONFIG = json.loads((CASE_ROOT / "config/case.json").read_text())
W, H = CONFIG["raster_size"]
BW, BH = CONFIG["board_size_m"]
PPM = W / BW
PALETTE=np.array([(244,245,238)]+[c["rgb"] for c in CONFIG["colors"]],dtype=np.uint8)
INK = tuple(PALETTE[1])
PAPER = (244, 245, 238)


def xy_to_pixel(xy):
    a = np.asarray(xy, dtype=float)
    return np.stack([(a[..., 0] + BW / 2) * PPM,
                     (BH / 2 - a[..., 1]) * PPM], axis=-1)


def pixel_to_xy(rc):
    a = np.asarray(rc, dtype=float)
    return np.stack([a[..., 1] / PPM - BW / 2,
                     BH / 2 - a[..., 0] / PPM], axis=-1)


def raster(paths):
    im = Image.new("L", (W, H))
    d = ImageDraw.Draw(im)
    width = round(CONFIG["pen_width_m"] * PPM)
    for path in paths:
        pts = [tuple(p) for p in xy_to_pixel(path)]
        d.line(pts, fill=255, width=width, joint="curve")
        for x, y in (pts[0], pts[-1]):
            r = width / 2 - 0.5
            d.ellipse((x-r, y-r, x+r, y+r), fill=255)
    return np.array(im) > 0


def board_rgb(mask):
    return PALETTE[np.asarray(mask,dtype=np.uint8)]


def color_layers(paths,colors):
    labels=np.zeros((H,W),dtype=np.uint8)
    for path,color in zip(paths,colors):labels[raster([path])]=color
    return labels


def circle(x, y, r):
    t = np.linspace(0, 2 * math.pi, 101)
    return np.stack([x + r*np.cos(t), y + r*np.sin(t)], axis=1)


def make_case(difficulty='hard', seed=0):
    raise ValueError('public runtime requires an explicitly admitted source_case; source generation is private')


def labels_from_rgb(rgb):
    values=np.asarray(rgb,dtype=np.int32)
    return np.argmin(np.sum((values[:,:,None,:]-PALETTE[None,None,:,:].astype(np.int32))**2,axis=-1,dtype=np.int64),axis=-1).astype(np.uint8)



def from_case(case):
    """Rasterize exact SHA-bound acquisition paths, avoiding platform float regeneration."""
    targets=case['target_paths_m'];additions=case['addition_paths_m']
    target=color_layers(targets,case['target_path_colors']);work=target.copy()
    for path,color in zip(additions,case['addition_path_colors']):work[raster([path])]=color
    return case,target,work
