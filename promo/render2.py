"""GUTROB — Шлёп, шлёп. Promo v2 (directed, 2.5D, events). 1080x1920 @60fps, 15s.
usage: render2.py WORKDIR OUT.mp4            -> full render
       render2.py WORKDIR OUTDIR t1,t2,...   -> preview jpgs
       render2.py WORKDIR OUTDIR seq:t0:t1   -> dense frame strip (every frame)
"""
import sys, math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
cv2.setNumThreads(1)

D = sys.argv[1]
W, H, FPS, DUR = 1080, 1920, 60, 15.0
T0 = 105.45
rng = np.random.default_rng(11)
def L(x): return x - T0

BEATS = [L(105.19 + 0.6035 * i) for i in range(27)]
SHLEP = [L(105.68), L(106.24)]            # ground stamps
HITS = [L(110.44), L(111.08)]             # big hits (letter drop + shockwave)
OUTRO = L(115.47)
GUT_T = L(116.68)
CTA_T = L(117.28)
COAL_ON, COAL_OUT = L(111.35), L(112.50)
IGNITE, BURN_END = L(113.40), L(114.95)

LINES = [
    ([("ухожу", 106.78), ("без", 107.10), ("оглядки", 107.36)], 108.05),
    ([("шлёпки", 108.10), ("стучат", 108.58), ("по", 109.04), ("пустой", 109.20), ("дорожке", 109.70)], 110.30),
    ([("угас", 111.48), ("последний", 111.68), ("уголь", 112.50)], 112.98),
    ([("а", 113.02), ("любовь", 113.08), ("сгорела,", 113.40), ("как", 113.98), ("сухой", 114.08), ("листок", 114.58)], 115.30),
]
LINES = [([(w, L(s)) for w, s in ws], L(e)) for ws, e in LINES]
LINE_BURN = (L(114.85), L(115.40))  # last line burns away

F_RUS = os.path.join(D, "fonts/RuslanDisplay.ttf")
F_ALE = os.path.join(D, "fonts/AlegreyaSC.ttf")
F_OSW = os.path.join(D, "fonts/Oswald.ttf")

def clamp01(x): return min(1.0, max(0.0, x))
def ease_io(x):
    x = clamp01(x); return x * x * (3 - 2 * x)
def ease_out(x):
    x = clamp01(x); return 1 - (1 - x) ** 3
def smooth_noise(h, w, cell, seed):
    r = np.random.default_rng(seed)
    n = r.random((max(2, h // cell + 2), max(2, w // cell + 2))).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    return (n - n.min()) / (n.max() - n.min() + 1e-6)

# ------------------------------------------------------------------ plate & layers
src = Image.open(os.path.join(D, "hero.jpg")).convert("RGB")
cut = Image.open(os.path.join(D, "hero_cut.png")).convert("RGBA")
a0 = np.array(cut)[:, :, 3]
mask = cv2.dilate((a0 > 20).astype(np.uint8) * 255, np.ones((25, 25), np.uint8))
plate0 = cv2.inpaint(np.array(src)[:, :, ::-1], mask, 21, cv2.INPAINT_TELEA)[:, :, ::-1]
plate0 = cv2.GaussianBlur(plate0, (0, 0), 1.0)
TOPX, BOTX = 250, 126
sw, sh = src.size
ref = plate0[0:50, 0:560].reshape(-1, 3)
col = np.median(ref, axis=0)
PH = sh + TOPX + BOTX
plate = np.zeros((PH, sw, 3), np.float32)
g = np.linspace(1.04, 1.0, TOPX + 60)[:, None, None]
plate[:TOPX + 60] = col[None, None, :] * g
plate[TOPX:TOPX + sh] = plate0
for i in range(60):
    a = 1 - i / 60
    plate[TOPX + i] = plate[TOPX + i] * (1 - a) + (col * g[TOPX + i, 0, 0]) * a
plate[TOPX + sh:] = cv2.GaussianBlur(plate0[sh - BOTX:][::-1].astype(np.float32), (0, 0), 5)
plate = np.clip(plate, 0, 255).astype(np.float32) / 255.0
hero = np.zeros((PH, sw, 4), np.float32)
ha = np.array(cut).astype(np.float32) / 255.0
ha[:, :, 3] = cv2.GaussianBlur(ha[:, :, 3], (0, 0), 0.9)
hero[TOPX:TOPX + sh] = ha
# premultiply for clean resampling
hero[:, :, :3] *= hero[:, :, 3:4]
FEET = (656.0, 975.0 + TOPX)

# heuristic depth (0 far .. 1 near)
yy, xx = np.mgrid[0:PH, 0:sw].astype(np.float32)
ground = np.clip((yy - 700) / 640, 0, 1) ** 0.9
trees = ((xx > 770) & (yy < 1050) & (yy > 300)).astype(np.float32) * 0.38
cliffs = ((xx < 560) & (yy > 690) & (yy < 1000)).astype(np.float32) * 0.22
depth = np.maximum(ground, np.maximum(cv2.GaussianBlur(trees, (0, 0), 25), cv2.GaussianBlur(cliffs, (0, 0), 25)))
depth = cv2.GaussianBlur(depth, (0, 0), 22).astype(np.float32)
D_HERO = float(depth[int(FEET[1]) - 20, int(FEET[0])])

S0 = H / PH
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

# ------------------------------------------------------------------ camera (Catmull-Rom on keys)
KEYS = np.array([
    # t,    cx,    cy,    z,    roll
    [-1.0, 400, 1030, 1.95, -2.5],
    [0.0, 420, 1012, 1.85, -2.0],
    [1.2, 480, 975, 1.62, -1.2],
    [2.5, 600, 770, 1.12, 0.4],
    [4.9, 640, 735, 1.30, 1.4],
    [6.1, 600, 790, 1.20, 0.0],
    [7.5, 585, 800, 1.24, -0.8],
    [9.0, 545, 760, 1.16, -1.4],
    [10.1, 585, 690, 1.08, -0.2],
    [12.5, 610, 690, 1.10, 0.7],
    [15.0, 622, 700, 1.14, 1.1],
    [16.0, 626, 704, 1.15, 1.2],
], np.float32)
def cam_raw(t):
    k = np.searchsorted(KEYS[:, 0], t) - 1
    k = int(min(max(k, 1), len(KEYS) - 3))
    p0, p1, p2, p3 = KEYS[k - 1], KEYS[k], KEYS[k + 1], KEYS[k + 2]
    u = (t - p1[0]) / (p2[0] - p1[0])
    # non-uniform: tangents scaled by interval
    m1 = (p2[1:] - p0[1:]) / (p2[0] - p0[0]) * (p2[0] - p1[0])
    m2 = (p3[1:] - p1[1:]) / (p3[0] - p1[0]) * (p2[0] - p1[0])
    h00 = 2 * u**3 - 3 * u**2 + 1; h10 = u**3 - 2 * u**2 + u; h01 = -2 * u**3 + 3 * u**2; h11 = u**3 - u**2
    return h00 * p1[1:] + h10 * m1 + h01 * p2[1:] + h11 * m2

def last_ev(t, evs):
    p = [e for e in evs if e <= t]
    return (t - p[-1], len(p) - 1) if p else (None, -1)

def camera(t):
    cx, cy, z, roll = [float(v) for v in cam_raw(t)]
    db, _ = last_ev(t, BEATS)
    beat = 0.012 * math.exp(-db * 10) if db is not None else 0
    dh, _ = last_ev(t, HITS + SHLEP + [OUTRO, GUT_T])
    hit = math.exp(-dh * 8) if dh is not None and dh < 1 else 0
    z *= 1 + beat + 0.045 * hit
    sc = S0 * z
    shake = 9 * hit
    cx += shake * math.sin(t * 97) / sc
    cy += shake * math.cos(t * 83) / sc
    roll += 0.7 * hit * math.sin(t * 61)
    # clamp so frame stays inside plate (with parallax margin)
    hw, hh = W / 2 / sc + 12, H / 2 / sc + 12
    cx = min(max(cx, hw), sw - hw)
    cy = min(max(cy, hh), PH - hh)
    parx = (cx - 600) * 0.22
    pary = (cy - 800) * 0.10
    return dict(cx=cx, cy=cy, z=z, sc=sc, roll=math.radians(roll), parx=parx, pary=pary, hit=hit)

def base_uv(c):
    X = gx - W / 2; Y = gy - H / 2
    cr, sr = math.cos(c["roll"]), math.sin(c["roll"])
    Xr = cr * X + sr * Y; Yr = -sr * X + cr * Y
    return c["cx"] + Xr / c["sc"], c["cy"] + Yr / c["sc"]

def dolly_q(c, d):
    return 1 + 0.32 * (c["z"] - 1.2) * d

def project(c, P, d=None):
    """plate point -> screen point (inverse of sampling map)."""
    px, py = P
    if d is None:
        d = float(depth[int(min(max(py, 0), PH - 1)), int(min(max(px, 0), sw - 1))])
    q = dolly_q(c, d)
    ux = c["cx"] + (px - c["parx"] * d - c["cx"]) * q
    uy = c["cy"] + (py - c["pary"] * d - c["cy"]) * q
    X = (ux - c["cx"]) * c["sc"]; Y = (uy - c["cy"]) * c["sc"]
    cr, sr = math.cos(c["roll"]), math.sin(c["roll"])
    return (W / 2 + cr * X - sr * Y, H / 2 + sr * X + cr * Y), q * c["sc"]

# ------------------------------------------------------------------ fog, grain, vignette
FOG_A = smooth_noise(H, 2600, 140, 1); FOG_A = cv2.GaussianBlur(FOG_A, (0, 0), 30)
FOG_B = smooth_noise(H, 2600, 200, 2); FOG_B = cv2.GaussianBlur(FOG_B, (0, 0), 45)
yv = np.linspace(0, 1, H, dtype=np.float32)[:, None]
GRAIN = [rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32) for _ in range(8)]
xv = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
yn = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
VIGN = np.clip(1 - 0.55 * ((xv**2) * 0.9 + (yn**2) * 0.6) ** 1.2, 0, 1)[:, :, None]

# ------------------------------------------------------------------ text helpers
def text_img(txt, font, size, fill=(240, 236, 228), stroke=0, stroke_fill=(20, 16, 12), shadow=True, glow=None, spacing=0):
    f = ImageFont.truetype(font, size)
    widths = [f.getlength(ch) for ch in txt]
    tw = int(sum(widths) + spacing * (len(txt) - 1)) if spacing else int(f.getlength(txt))
    asc, desc = f.getmetrics(); pad = size // 2
    im = Image.new("RGBA", (tw + pad * 2, asc + desc + pad * 2), (0, 0, 0, 0))
    def draw(dr, colr, off=(0, 0), sw_=stroke, sf=stroke_fill):
        if spacing:
            x = pad
            for ch, cw in zip(txt, widths):
                dr.text((x + off[0], pad + off[1]), ch, font=f, fill=colr, stroke_width=sw_, stroke_fill=sf); x += cw + spacing
        else:
            dr.text((pad + off[0], pad + off[1]), txt, font=f, fill=colr, stroke_width=sw_, stroke_fill=sf)
    if shadow:
        s = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw(ImageDraw.Draw(s), (0, 0, 0, 210), (0, size // 16), stroke + 2, (0, 0, 0, 210))
        im.alpha_composite(s.filter(ImageFilter.GaussianBlur(size / 10)))
    if glow:
        gl = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw(ImageDraw.Draw(gl), glow, (0, 0), stroke + 4, glow)
        im.alpha_composite(gl.filter(ImageFilter.GaussianBlur(size / 6)))
    draw(ImageDraw.Draw(im), fill)
    return im

def letters(txt, font, size, **kw):
    f = ImageFont.truetype(font, size)
    imgs, xs, x = [], [], 0.0
    sp = kw.pop("spacing", 0)
    for ch in txt:
        cw = f.getlength(ch)
        imgs.append(text_img(ch, font, size, **kw) if ch.strip() else None)
        xs.append(x + cw / 2); x += cw + sp
    total = x - sp
    return imgs, [v - total / 2 for v in xs]

def paste(canvas, im, cx, cy, sx=1.0, sy=None, rot=0.0, alpha=1.0):
    if im is None or alpha <= 0.01: return
    sy = sx if sy is None else sy
    if sx != 1.0 or sy != 1.0:
        im = im.resize((max(1, int(im.width * sx)), max(1, int(im.height * sy))), Image.BICUBIC)
    if rot: im = im.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 1:
        im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

# ground stamp texture (white mask; colored by temperature at runtime)
STAMP_F = ImageFont.truetype(F_RUS, 220)
def stamp_mask(txt):
    tw = int(STAMP_F.getlength(txt)); asc, desc = STAMP_F.getmetrics()
    im = Image.new("L", (tw + 40, asc + desc + 30), 0)
    ImageDraw.Draw(im).text((20, 10), txt, font=STAMP_F, fill=255)
    return np.asarray(im).astype(np.float32) / 255.0
GSTAMP = stamp_mask("ШЛЁП")
GS_QUADS = [  # plate-space ground quads: center, half-width, half-depth, angle(deg)
    ((380, 1168), 128, 42, -7),
    ((505, 1232), 140, 46, 5),
]
def temp_color(T):
    # T 1 white-hot -> 0 charcoal
    stops = [(0.0, (0.10, 0.08, 0.07)), (0.25, (0.45, 0.06, 0.02)), (0.5, (0.95, 0.30, 0.04)), (0.8, (1.0, 0.70, 0.25)), (1.0, (1.0, 0.95, 0.80))]
    for (t1, c1), (t2, c2) in zip(stops, stops[1:]):
        if T <= t2:
            u = (T - t1) / (t2 - t1); return np.array(c1) * (1 - u) + np.array(c2) * u
    return np.array(stops[-1][1])

# lyrics
WSIZE = 86
WCOL, WHOT = (238, 232, 222), (255, 176, 80)
LINE_IMGS = []
for ws, end in LINES:
    items = []
    for w, s in ws:
        items.append((text_img(w.upper(), F_ALE, WSIZE, fill=WCOL, stroke=2),
                      text_img(w.upper(), F_ALE, WSIZE, fill=WHOT, stroke=2, glow=(255, 110, 20, 150)), s))
    LINE_IMGS.append((items, ws[0][1], end))
def layout(items, maxw=960):
    sp, pad = 24, WSIZE // 2
    rows, cur, cw = [], [], 0
    for i, (im, _, _) in enumerate(items):
        ww = im.width - 2 * pad
        if cur and cw + sp + ww > maxw:
            rows.append((cur, cw)); cur, cw = [], 0
        cur.append((i, cw + (sp if cur else 0))); cw += (sp if len(cur) > 1 else 0) + ww
    rows.append((cur, cw)); return rows
LAYOUT = [layout(it) for it, _, _ in LINE_IMGS]
LY = 1240

TOP = text_img("GUTROB", F_RUS, 60, fill=(235, 230, 220), spacing=14)
DROP_L, DROP_X = letters("ШЛЁП", F_RUS, 230, fill=(246, 238, 224), stroke=3, glow=(255, 130, 30, 160))
GUT_L, GUT_X = letters("GUTROB", F_RUS, 118, fill=(236, 230, 220), spacing=20)
CTA_TXT = "СЛУШАЙ ПО ССЫЛКЕ В ШАПКЕ"
CTA_L, CTA_X = letters(CTA_TXT, F_OSW, 50, fill=(232, 226, 216), spacing=4)

# title (for ember convergence): two rows
TITLE_ROWS = [("ШЛЁП,", 575, -3), ("ШЛЁП", 775, 3)]
TITLE_SIZE = 205
TITLE_HOT = [text_img(s, F_RUS, TITLE_SIZE, fill=(255, 214, 150), stroke=3, glow=(255, 120, 20, 200)) for s, _, _ in TITLE_ROWS]
TITLE_COOL = [text_img(s, F_RUS, TITLE_SIZE, fill=(246, 238, 224), stroke=3, glow=(255, 130, 30, 90)) for s, _, _ in TITLE_ROWS]
# target points for embers
tpts = []
for (s, y, rot), im in zip(TITLE_ROWS, TITLE_COOL):
    a = np.asarray(im.rotate(rot, Image.BICUBIC, expand=True).getchannel("A"))
    ys, xs = np.nonzero(a > 200)
    sel = rng.choice(len(xs), 520, replace=False)
    hh, ww = a.shape
    tpts.append(np.stack([xs[sel] - ww / 2 + W / 2, ys[sel] - hh / 2 + y], 1))
TPTS = np.concatenate(tpts).astype(np.float32)
NT = len(TPTS)

# ------------------------------------------------------------------ coal
def make_coal(sz=220):
    yy_, xx_ = np.mgrid[0:sz, 0:sz].astype(np.float32)
    cx_, cy_ = sz / 2, sz / 2
    ang = np.arctan2(yy_ - cy_, xx_ - cx_)
    rad = np.hypot((xx_ - cx_) / 1.0, (yy_ - cy_) / 0.72)
    r_edge = sz * 0.40 * (1 + 0.12 * np.sin(ang * 3 + 1) + 0.07 * np.sin(ang * 7 + 2))
    alpha = np.clip((r_edge - rad) / 3, 0, 1)
    n1 = smooth_noise(sz, sz, 18, 5); n2 = smooth_noise(sz, sz, 7, 6)
    ridge = 1 - np.abs(n1 * 2 - 1)
    cracks = np.clip((ridge - 0.90) / 0.07, 0, 1) * (0.6 + 0.4 * n2)
    shade = np.clip(1.15 - (yy_ - cy_ + 0.4 * (xx_ - cx_)) / (sz * 0.6), 0.4, 1.3)
    rock = np.stack([0.16, 0.14, 0.13], -1)[None, None, :] * (0.6 + 0.5 * n2[:, :, None]) * shade[:, :, None]
    return rock.astype(np.float32), cracks.astype(np.float32), alpha.astype(np.float32)
COAL_ROCK, COAL_CRK, COAL_A = make_coal()
COAL_P = (440, 1212)
def coal_I(t):
    if t < COAL_ON: return 0.0
    i = ease_out((t - COAL_ON) / 0.35)
    fl = 0.82 + 0.18 * math.sin(t * 23) * math.sin(t * 7.3)
    if t > COAL_OUT: i *= max(0, 1 - (t - COAL_OUT) / 0.35) ** 1.5
    return i * fl
def coal_vis(t):
    if t < COAL_ON - 0.05: return 0.0
    v = ease_out((t - COAL_ON + 0.05) / 0.3)
    if t > IGNITE + 0.6: v *= max(0, 1 - (t - IGNITE - 0.6) / 0.6)
    return v

# ------------------------------------------------------------------ leaf
LW, LH = 600, 330
def make_leaf():
    yy_, xx_ = np.mgrid[0:LH, 0:LW].astype(np.float32)
    x = xx_ / LW; cy_ = LH / 2
    n = smooth_noise(LH, LW, 12, 9)
    half = 0.5 * LH * 0.9 * np.clip(np.sin(np.pi * np.clip((x - 0.04) / 0.96, 0, 1)), 0, 1) ** 0.75 * (1 - 0.3 * x)
    half *= 1 + 0.07 * np.sin(x * 46) + 0.05 * (n - 0.5)
    curve = cy_ + 18 * np.sin(x * np.pi) - 10
    dy = np.abs(yy_ - curve)
    alpha = np.clip((half - dy) / 2, 0, 1)
    # stem
    stem = (x < 0.07) & (np.abs(yy_ - curve) < 3.5)
    alpha = np.maximum(alpha, stem.astype(np.float32))
    mott = smooth_noise(LH, LW, 30, 10)
    base = np.array([0.62, 0.38, 0.14]) * (0.75 + 0.45 * mott[:, :, None])
    edge = np.clip(dy / (half + 1e-3), 0, 1)[:, :, None]
    colr = base * (1 - 0.35 * edge) + np.array([0.35, 0.18, 0.07]) * 0.35 * edge
    # veins
    vein = np.zeros((LH, LW), np.float32)
    vein[np.abs(yy_ - curve) < 2.2] = 1
    for k in range(1, 10):
        x0 = 0.08 + k * 0.085
        for sgn in (-1, 1):
            # line from (x0, curve) towards tip and edge
            t_ = np.linspace(0, 1, 120)
            px = (x0 + 0.12 * t_) * LW
            py = np.interp(px, xx_[0], curve[0]) + sgn * t_ * np.interp(px, xx_[0], half.max(0) if False else half[int(cy_)]) * 0.9
            for a_, b_ in zip(px.astype(int), py.astype(int)):
                if 0 <= a_ < LW and 0 <= b_ < LH:
                    vein[max(0, b_ - 1):b_ + 1, a_] = 0.8
    vein = cv2.GaussianBlur(vein, (0, 0), 0.8)
    colr = colr * (1 - 0.4 * vein[:, :, None])
    shade = 0.85 + 0.3 * (1 - edge)
    colr = np.clip(colr * shade, 0, 1)
    burnN = 0.55 * (1 - x) + 0.45 * smooth_noise(LH, LW, 40, 12)
    burnN = (burnN - burnN[alpha > 0.5].min()) / (burnN[alpha > 0.5].max() - burnN[alpha > 0.5].min())
    return colr.astype(np.float32), alpha.astype(np.float32), burnN.astype(np.float32)
LEAF_C, LEAF_A, LEAF_N = make_leaf()
LEAF_IN, LEAF_FREEZE = L(112.65), L(114.05)
def leaf_pose(t):
    u = clamp01((t - LEAF_IN) / (LEAF_FREEZE - LEAF_IN))
    e = 1 - (1 - u) ** 3
    x = 1250 - 990 * e + 30 * math.sin(t * 2.1)
    y = 760 - 90 * e + 50 * math.sin(t * 3.0 + 0.5) - 90 * clamp01((t - IGNITE) / 1.6) ** 2
    ang = 25 * math.sin(t * 1.7) - 20 * e
    flip = math.cos(t * 2.6 + 0.3)
    flip = math.copysign(max(abs(flip), 0.18), flip)
    s = 0.86 + 0.08 * math.sin(t * 1.3)
    return x, y, ang, flip, s
def burn_th(t):
    return -0.05 + 1.12 * clamp01((t - IGNITE) / (BURN_END - IGNITE)) ** 1.4

def leaf_point(t, lx, ly):
    x, y, ang, flip, s = leaf_pose(t)
    vx = (lx - LW / 2) * s * flip; vy = (ly - LH / 2) * s
    a = math.radians(ang)
    return x + vx * math.cos(a) + vy * math.sin(a), y - vx * math.sin(a) + vy * math.cos(a)

# ------------------------------------------------------------------ embers (analytic)
# A: leaf embers born on burn front
NA = 420
EA_tb = np.sort(rng.uniform(IGNITE + 0.05, BURN_END, NA))
EA_p0 = np.zeros((NA, 2), np.float32)
leaf_valid = LEAF_A > 0.6
for i, tb in enumerate(EA_tb):
    th = burn_th(tb)
    cand = np.argwhere(leaf_valid & (np.abs(LEAF_N - th) < 0.025))
    if len(cand) == 0: cand = np.argwhere(leaf_valid)
    yy_, xx_ = cand[rng.integers(len(cand))]
    EA_p0[i] = leaf_point(tb, xx_, yy_)
EA_v = np.stack([rng.uniform(-60, 60, NA), rng.uniform(-260, -110, NA)], 1).astype(np.float32)
EA_life = rng.uniform(0.9, 2.2, NA)
EA_sz = rng.uniform(1.6, 4.2, NA)
EA_ph = rng.uniform(0, 6.28, NA)
# B: title embers: rise from bottom/leaf area then converge
NB = NT
EB_tb = rng.uniform(L(114.3), L(115.1), NB)
EB_p0 = np.stack([rng.uniform(120, 960, NB), rng.uniform(1250, 1900, NB)], 1).astype(np.float32)
# a share is born from the leaf
k_ = NB // 3
for i in range(k_):
    j = rng.integers(NA); EB_p0[i] = EA_p0[j]; EB_tb[i] = EA_tb[j]
EB_v = np.stack([rng.uniform(-50, 50, NB), rng.uniform(-420, -200, NB)], 1).astype(np.float32)
EB_ph = rng.uniform(0, 6.28, NB)
EB_sz = rng.uniform(1.8, 3.6, NB)
CONV0 = OUTRO - 0.42
EB_conv = rng.uniform(0, 0.12, NB)  # stagger

# ash flakes always
NF = 110
AF = dict(x=rng.uniform(0, W, NF), y=rng.uniform(0, H, NF), sp=rng.uniform(25, 75, NF), sz=rng.uniform(1.2, 3.2, NF), ph=rng.uniform(0, 6.28, NF))

# smoke (stamps & coal) analytic puffs
def smoke_puffs():
    puffs = []
    for st, (P, hw, hd, angd) in zip(SHLEP, GS_QUADS):
        for k in range(14):
            puffs.append(("g", st + rng.uniform(0.05, 1.8), P[0] + rng.uniform(-hw * 0.8, hw * 0.8), P[1] + rng.uniform(-hd, hd), rng.uniform(1.2, 2.2)))
    for k in range(18):
        puffs.append(("g", COAL_OUT + rng.uniform(0.0, 1.1), COAL_P[0] + rng.uniform(-15, 15), COAL_P[1] - 10, rng.uniform(1.4, 2.4)))
    return puffs
PUFFS = smoke_puffs()

# dust bursts at stamp contacts (screen-space after projection)
DUST = [(st, rng.uniform(0, 6.28, 40), rng.uniform(150, 520, 40), rng.uniform(3, 9, 40)) for st in SHLEP]

# ------------------------------------------------------------------ grading
def grade(arr, t, sat):
    lum = (arr[:, :, 0] * 0.3 + arr[:, :, 1] * 0.59 + arr[:, :, 2] * 0.11)[:, :, None]
    arr = lum + (arr - lum) * sat
    arr = np.clip((arr - 0.5) * 1.18 + 0.47, 0, 1)
    arr = arr + np.array([-0.02, 0.0, 0.035], np.float32) * (1 - lum) + np.array([0.03, 0.012, -0.01], np.float32) * lum
    return arr

def sat_curve(t):
    s = 0.86
    if t > COAL_OUT: s -= 0.38 * ease_io((t - COAL_OUT) / 0.45)
    if t > IGNITE: s += 0.16 * ease_io((t - IGNITE) / 1.2)
    if t > OUTRO: s += 0.12 * ease_io((t - OUTRO) / 0.8)
    return s

def dof_blur(t):
    b = 0.0
    if t > COAL_ON - 0.2: b = 4.0 * ease_io((t - COAL_ON + 0.2) / 0.5)
    if t > LEAF_IN: b = 4.0 - 1.2 * ease_io((t - LEAF_IN) / 0.6)
    if t > OUTRO - 0.3: b = 2.8 * (1 - ease_io((t - OUTRO + 0.3) / 0.6))
    return b

# ------------------------------------------------------------------ frame
def remap_rgb(img, mx, my):
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def frame(t):
    c = camera(t)
    ux, uy = base_uv(c)
    dmap = cv2.remap(depth, ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    q = dolly_q(c, dmap)
    sx_ = c["cx"] + (ux - c["cx"]) / q + c["parx"] * dmap
    sy_ = c["cy"] + (uy - c["cy"]) / q + c["pary"] * dmap
    arr = remap_rgb(plate, sx_, sy_)

    # back fog band (mid depth)
    off = int((t * 38 + c["parx"] * 0.3) % 1500)
    band = np.clip(1 - np.abs(yv - (0.50 - (c["cy"] - 800) / 2400)) / 0.28, 0, 1)
    fb = FOG_A[:, off:off + W] * band
    arr = arr * (1 - 0.32 * fb[:, :, None]) + 0.84 * 0.32 * fb[:, :, None]

    # hero layer
    k = 1.0 + 0.085 * ease_out(t / 12.5)
    db, bi = last_ev(t, BEATS)
    step = math.sin(min(db / 0.6035, 1) * math.pi) if db is not None else 0
    side = 1 if bi % 2 == 0 else -1
    bob = 5.5 * step + 8 * c["hit"]
    tilt = math.radians(0.9 * side * step)
    fx, fy = FEET
    qh = dolly_q(c, D_HERO)
    # screen->plate (same as bg with d=D_HERO), then plate->hero-local (scale k, bob, tilt about feet)
    px_ = c["cx"] + (ux - c["cx"]) / qh + c["parx"] * D_HERO
    py_ = c["cy"] + (uy - c["cy"]) / qh + c["pary"] * D_HERO
    ct, st_ = math.cos(tilt), math.sin(tilt)
    rx, ry = px_ - fx, py_ - (fy - bob)
    hx = fx + (ct * rx + st_ * ry) / k
    hy = fy + (-st_ * rx + ct * ry) / k
    hl = cv2.remap(hero, hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ha_ = hl[:, :, 3:4]
    light = 1.0
    ci = coal_I(t) * coal_vis(t)
    arr = arr * (1 - ha_) + hl[:, :, :3] * (1 + 0.10 * c["hit"]) * light
    feet_scr, fsc = project(c, (fx, fy), D_HERO)

    # ground stamps (hot prints) — in world, on stones
    for st, (P, hw, hd, angd) in zip(SHLEP, GS_QUADS):
        dt = t - st
        if dt < -0.16 or dt > 4.6: continue
        a = math.radians(angd)
        corners = []
        for ox, oy in [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]:
            pxw = P[0] + ox * math.cos(a) - oy * math.sin(a) * 0.4
            pyw = P[1] + ox * math.sin(a) * 0.4 + oy
            (sx2, sy2), _ = project(c, (pxw, pyw))
            corners.append((sx2, sy2))
        corners = np.array(corners, np.float32)
        lift = 0.0; scl = 1.0; mb = 0
        if dt < 0:
            u = 1 + dt / 0.16  # 0..1
            lift = 420 * (1 - u) ** 2; scl = 1 + 0.5 * (1 - u); mb = int(30 * (1 - u)) + 1
            cen = corners.mean(0)
            corners = (corners - cen) * scl + cen - np.array([0, lift], np.float32)
        hgt, wid = GSTAMP.shape
        M = cv2.getPerspectiveTransform(np.array([[0, 0], [wid, 0], [wid, hgt], [0, hgt]], np.float32), corners)
        m = cv2.warpPerspective(GSTAMP, M, (W, H), flags=cv2.INTER_LINEAR)
        if mb > 1:
            kk = np.zeros((mb, 1), np.float32); kk[:] = 1 / mb
            m = cv2.filter2D(m, -1, kk)
        if dt < 0:
            colr = np.array([0.96, 0.93, 0.88]); alpha = 0.95; glow_amt = 0.2
        else:
            T = max(0.0, 1 - dt / 3.2) ** 0.8
            colr = temp_color(T); alpha = 0.92 if dt < 3.6 else 0.92 * max(0, 1 - (dt - 3.6) / 1.0)
            glow_amt = 1.1 * T
            # dark scorch underlay
            arr = arr * (1 - 0.45 * alpha * m[:, :, None])
        arr = arr * (1 - alpha * m[:, :, None]) + colr[None, None, :] * alpha * m[:, :, None] * (1 if dt < 0 else 1.0)
        if glow_amt > 0.02:
            gm = cv2.GaussianBlur(m, (0, 0), 18)
            arr = arr + gm[:, :, None] * np.array([1.0, 0.45, 0.1], np.float32) * glow_amt * 0.8 * alpha
        if 0 <= dt < 0.12:
            arr = arr + (1 - arr) * 0.35 * (1 - dt / 0.12)

    # coal on stones
    cv_ = coal_vis(t)
    if cv_ > 0.01:
        (csx, csy), csc = project(c, COAL_P)
        s_ = csc / 1.5
        I = ci
        rock = COAL_ROCK * (1 + 0.6 * I * COAL_CRK[:, :, None])
        hotc = np.array([1.0, 0.42 + 0.25 * I, 0.08], np.float32)
        cimg = rock * (1 - COAL_CRK[:, :, None] * I) + hotc * COAL_CRK[:, :, None] * (0.4 + 1.1 * I) * I
        dead = 1 - I
        cimg = cimg + COAL_CRK[:, :, None] * np.array([0.25, 0.24, 0.23]) * dead * (t > COAL_OUT) * 0.6
        sz = max(8, int(COAL_A.shape[0] * s_))
        cimg_r = cv2.resize(np.clip(cimg, 0, 2), (sz, sz)); ca_r = cv2.resize(COAL_A, (sz, sz)) * cv_
        x0, y0 = int(csx - sz / 2), int(csy - sz / 2)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + sz), min(H, y0 + sz)
        if xb > xa and yb > ya:
            sub = arr[ya:yb, xa:xb]
            cr_ = cimg_r[ya - y0:yb - y0, xa - x0:xb - x0]; ar_ = ca_r[ya - y0:yb - y0, xa - x0:xb - x0, None]
            # contact shadow
            arr[ya:yb, xa:xb] = sub * (1 - ar_) + np.minimum(cr_, 1.6) * ar_
            if I > 0.01:
                halo = cv2.GaussianBlur(ca_r * I, (0, 0), sz / 6)[ya - y0:yb - y0, xa - x0:xb - x0, None]
                arr[ya:yb, xa:xb] += halo * np.array([1.0, 0.4, 0.08], np.float32) * 0.6
        # warm light pool + glow
        if I > 0.01:
            d2 = ((gx - csx) ** 2 + ((gy - csy) * 1.6) ** 2)
            pool = np.exp(-d2 / (2 * (sz * 1.5) ** 2))[:, :, None]
            arr = arr + pool * np.array([1.0, 0.42, 0.10], np.float32) * 0.20 * I
            arr = arr * (1 + pool * 0.12 * I)

    # front fog (near, fast parallax)
    off2 = int((t * 75 + c["parx"] * 1.2) % 1500)
    ff = FOG_B[:, off2:off2 + W] * (np.clip((yv - 0.66) / 0.34, 0, 1) ** 1.4)
    arr = arr * (1 - 0.42 * ff[:, :, None]) + 0.86 * 0.42 * ff[:, :, None]

    # shockwave fog ring from feet on big hits
    for hk in HITS:
        dt = t - hk
        if 0 <= dt < 0.8:
            r = 60 + 1100 * ease_out(dt / 0.8)
            ring = np.exp(-((np.sqrt((gx - feet_scr[0]) ** 2 + ((gy - feet_scr[1]) * 3.6) ** 2) - r) ** 2) / (2 * (28 + 50 * dt) ** 2))
            arr = arr + (0.9 - arr) * ring[:, :, None] * 0.55 * (1 - dt / 0.8)

    # DOF: blur scene behind foreground objects
    b = dof_blur(t)
    if b > 0.25:
        arr = cv2.GaussianBlur(arr, (0, 0), b)
    # camera motion blur (vertical/zoom component)
    c2 = camera(t + 1 / 120)
    vy = (c2["cy"] - c["cy"]) * c["sc"]; vx = (c2["cx"] - c["cx"]) * c["sc"]
    ln = math.hypot(vx, vy)
    if ln > 1.2:
        n_ = int(min(ln, 24)) | 1
        kk = np.zeros((n_, n_), np.float32)
        ang = math.atan2(vy, vx)
        for i in range(n_):
            o = i - n_ // 2
            kk[int(n_ // 2 + o * math.sin(ang)), int(n_ // 2 + o * math.cos(ang))] = 1
        kk /= kk.sum(); arr = cv2.filter2D(arr, -1, kk)

    # smoke puffs
    sm = np.zeros((H // 2, W // 2), np.float32)
    for kind, tb, px, py, life in PUFFS:
        dt = t - tb
        if 0 <= dt < life:
            (sx2, sy2), scp = project(c, (px, py))
            u = dt / life
            sx2 += 40 * math.sin(dt * 2 + px) ; sy2 -= 260 * dt
            r = (18 + 90 * u) * scp / 2.2
            cv2.circle(sm, (int(sx2 / 2), int(sy2 / 2)), max(1, int(r / 2)), 0.22 * (1 - u) * math.sin(min(1, u * 6) * math.pi / 2), -1)
    if sm.max() > 0:
        sm = cv2.GaussianBlur(sm, (0, 0), 14)
        sm = cv2.resize(sm, (W, H))[:, :, None]
        arr = arr * (1 - sm) + 0.75 * sm

    # leaf
    if LEAF_IN <= t < BURN_END + 0.1:
        x, y, ang, flip, s = leaf_pose(t)
        th = burn_th(t)
        la = LEAF_A.copy()
        lc = LEAF_C.copy()
        if th > -0.05:
            burnt = LEAF_N < th
            la = la * (~burnt)
            charz = np.clip(1 - (LEAF_N - th) / 0.09, 0, 1) * (~burnt)
            lc = lc * (1 - 0.85 * charz[:, :, None])
            edge = np.clip(1 - np.abs(LEAF_N - th) / 0.018, 0, 1) * (LEAF_A > 0.3)
            lc = lc + edge[:, :, None] * np.array([1.4, 0.65, 0.15]) * (0.8 + 0.2 * math.sin(t * 40))
            la = np.maximum(la, edge * 0.95)
        if flip < 0:
            lc = lc * 0.72
        rgba = np.dstack([np.clip(lc, 0, 1.5) * la[:, :, None], la]).astype(np.float32)
        wsc, hsc = abs(flip) * s, s
        M = cv2.getRotationMatrix2D((LW / 2, LH / 2), ang, 1.0)
        A = np.array([[wsc * (1 if flip > 0 else -1), 0, 0], [0, hsc, 0]], np.float32)
        # compose: local -> scale/flip about center -> rotate -> translate to (x,y)
        cen = np.array([LW / 2, LH / 2])
        R = np.array([[math.cos(math.radians(ang)), math.sin(math.radians(ang))], [-math.sin(math.radians(ang)), math.cos(math.radians(ang))]])
        Sm = np.diag([wsc * (1 if flip > 0 else -1), hsc])
        Mx = R @ Sm
        tvec = np.array([x, y]) - Mx @ cen
        MA = np.hstack([Mx, tvec[:, None]]).astype(np.float32)
        lw = cv2.warpAffine(rgba, MA, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        # leaf motion blur (along travel)
        al = lw[:, :, 3:4]
        arr = arr * (1 - al) + lw[:, :, :3]
        if th > 0:
            gl = cv2.GaussianBlur(lw[:, :, 3] * (th < 1.0), (0, 0), 30)
            arr = arr + gl[:, :, None] * np.array([1.0, 0.45, 0.1], np.float32) * 0.35 * clamp01(th * 3) * (1 - clamp01((t - BURN_END + 0.3) / 0.4))

    # particles layer (half-res) : ash, leaf embers, title embers
    pl = np.zeros((H // 2, W // 2, 3), np.float32)
    for i in range(NF):
        yy2 = (AF["y"][i] + AF["sp"][i] * t) % H; xx2 = AF["x"][i] + 16 * math.sin(t * 0.8 + AF["ph"][i]) - c["parx"] * 0.6
        cv2.circle(pl, (int(xx2 / 2) % (W // 2), int(yy2 / 2)), max(1, int(AF["sz"][i] / 2)), (0.5, 0.5, 0.52), -1)
    m = (t >= EA_tb) & (t < EA_tb + EA_life)
    for i in np.nonzero(m)[0]:
        dt = t - EA_tb[i]; u = dt / EA_life[i]
        p = EA_p0[i] + EA_v[i] * dt + np.array([30 * math.sin(dt * 3 + EA_ph[i]), -40 * dt * dt])
        fl = (1 - u) * (0.7 + 0.3 * math.sin(t * 30 + EA_ph[i]))
        cv2.circle(pl, (int(p[0] / 2), int(p[1] / 2)), max(1, int(EA_sz[i] / 2 * (1 - 0.5 * u))), (1.0 * fl * 1.6, 0.5 * fl * 1.6, 0.12 * fl), -1)
    if t >= EB_tb.min():
        for i in range(NB):
            dt = t - EB_tb[i]
            if dt < 0: continue
            free = EB_p0[i] + EB_v[i] * dt + np.array([40 * math.sin(dt * 2.2 + EB_ph[i]), 0])
            w_ = ease_io((t - CONV0 - EB_conv[i]) / (OUTRO - CONV0 - EB_conv[i])) if t > CONV0 + EB_conv[i] else 0
            p = free * (1 - w_) + TPTS[i] * w_
            fade = 1.0 if t < OUTRO else max(0, 1 - (t - OUTRO) / 0.45)
            if fade <= 0: continue
            fl = fade * (0.75 + 0.25 * math.sin(t * 25 + EB_ph[i]))
            cv2.circle(pl, (int(p[0] / 2), int(p[1] / 2)), max(1, int(EB_sz[i] / 2)), (1.0 * fl * 1.5, 0.55 * fl * 1.5, 0.15 * fl), -1)
    glow = cv2.GaussianBlur(pl, (0, 0), 3.5)
    pl = cv2.resize(pl + glow * 1.4, (W, H))
    arr = arr + pl  # additive light

    # grade
    arr = grade(np.clip(arr, 0, 1.4), t, sat_curve(t))
    if t > COAL_OUT:  # colder after coal dies
        arr = arr + np.array([-0.025, -0.005, 0.03], np.float32) * ease_io((t - COAL_OUT) / 0.5) * (1 - 0.6 * clamp01((t - IGNITE) / 1.5))
    arr = arr * VIGN
    gr = cv2.resize(GRAIN[int(t * 24) % 8], (W, H), interpolation=cv2.INTER_NEAREST)
    arr = arr + gr[:, :, None] * 0.026
    if t > OUTRO:
        arr = arr * (1 - 0.30 * ease_io((t - OUTRO) / 0.5))
    fade = clamp01((DUR - t) / 0.7)
    arr = arr * fade
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).convert("RGBA")

    # ---------------- typography
    if 1.15 < t < LEAF_IN + 0.2:
        a_ = ease_out((t - 1.15) / 0.4) * clamp01((LEAF_IN + 0.2 - t) / 0.3)
        paste(img, TOP, W / 2, 225, alpha=0.8 * a_)
    # big hits: letters drop with weight
    for j, hk in enumerate(HITS):
        base_y = 820 if j == 0 else 1010
        base_x = W / 2 + (-40 if j == 0 else 40)
        rot = 5 if j == 0 else -5
        for li, (im, lx) in enumerate(zip(DROP_L, DROP_X)):
            tl = hk + li * 0.028
            dt = t - tl
            if dt < -0.16 or t - hk > 0.62: continue
            if dt < 0:
                u = 1 + dt / 0.16; y = base_y - 380 * (1 - u) ** 2; sx, sy = 0.92, 1.12; al = clamp01(u * 3)
            else:
                sq = math.exp(-dt * 16) * math.cos(dt * 30)
                y = base_y; sx, sy = 1 + 0.16 * sq, 1 - 0.18 * sq; al = 1.0
            if t - hk > 0.42: al *= max(0, 1 - (t - hk - 0.42) / 0.2)
            ang = math.radians(rot)
            paste(img, im, base_x + lx * math.cos(ang), y - lx * math.sin(ang) + (sy - 1) * -60, sx, sy, rot=rot, alpha=al)
    # lyrics
    for li, (items, start, end) in enumerate(LINE_IMGS):
        pre0 = start - 0.2 if li == 0 else max(start - 0.2, LINE_IMGS[li - 1][2] + 0.14)
        if not (min(pre0, start - 0.12) <= t < end + 0.14): continue
        burning = li == 3
        aout = 1.0 if (t < end or burning) else max(0, 1 - (t - end) / 0.12)
        rows = LAYOUT[li]
        base_y = LY - (len(rows) - 1) * 56
        line_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0)) if burning else img
        for ri, (row, rw) in enumerate(rows):
            y = base_y + ri * 112
            x0 = W / 2 - rw / 2
            for wi, xo in row:
                norm, hot, ws = items[wi]
                pad = WSIZE // 2
                cxw = x0 + xo + (norm.width - 2 * pad) / 2
                wt = t - ws
                if wt < -0.12:
                    if t < pre0: continue
                    pre = ease_out((t - pre0) / 0.25)
                    paste(line_layer, norm, cxw, y + 14, 0.96, alpha=0.30 * pre * aout)
                else:
                    u = clamp01((wt + 0.12) / 0.12)
                    if u < 1:
                        sc = 1.5 - 0.5 * ease_out(u); yo = -40 * (1 - u); al = u
                    else:
                        sq = math.exp(-(wt) * 14) * math.cos(wt * 28)
                        sc = 1 + 0.07 * sq; yo = 0; al = 1
                    im = hot if 0 <= wt < 0.5 else norm
                    paste(line_layer, im, cxw, y + yo, sc, alpha=al * aout)
        if burning:
            la = np.asarray(line_layer).astype(np.float32) / 255.0
            b0, b1 = LINE_BURN
            if t > b0:
                th = 1.1 * clamp01((t - b0) / (b1 - b0))
                N = BURN_TXT_N
                alive = (N > th).astype(np.float32)
                edge = np.clip(1 - np.abs(N - th) / 0.03, 0, 1) * (la[:, :, 3] > 0.1)
                la[:, :, 3] *= alive
                la[:, :, :3] = la[:, :, :3] * (1 - edge[:, :, None]) + np.array([1.0, 0.6, 0.15]) * edge[:, :, None]
                la[:, :, 3] = np.maximum(la[:, :, 3], edge * 0.9)
            img.alpha_composite(Image.fromarray((np.clip(la, 0, 1) * 255).astype(np.uint8)))
    # title forms from embers
    if t >= OUTRO - 0.02:
        dt = t - OUTRO
        fo = clamp01((DUR - t) / 0.7)
        for k_, ((s, y, rot), hot, cool) in enumerate(zip(TITLE_ROWS, TITLE_HOT, TITLE_COOL)):
            a = ease_out((dt + 0.02) / 0.14) * fo
            cool_u = clamp01((dt - 0.15) / 0.9)
            db2, _ = last_ev(t, BEATS)
            pul = 1 + 0.025 * math.exp(-db2 * 9) if db2 is not None else 1
            if cool_u < 1: paste(img, hot, W / 2, y, pul, rot=rot, alpha=a * (1 - cool_u))
            if cool_u > 0: paste(img, cool, W / 2, y, pul, rot=rot, alpha=a * cool_u)
        # GUTROB letters drop
        for li, (im, lx) in enumerate(zip(GUT_L, GUT_X)):
            tl = GUT_T + li * 0.035; d2 = t - tl
            if im is None or d2 < -0.14: continue
            if d2 < 0:
                u = 1 + d2 / 0.14; y = 975 - 300 * (1 - u) ** 2; sx, sy = 0.94, 1.1; al = clamp01(u * 3)
            else:
                sq = math.exp(-d2 * 15) * math.cos(d2 * 28); y = 975; sx, sy = 1 + 0.14 * sq, 1 - 0.16 * sq; al = 1
            paste(img, im, W / 2 + lx, y, sx, sy, alpha=al * fo)
        # CTA type-on
        for li, (im, lx) in enumerate(zip(CTA_L, CTA_X)):
            tl = CTA_T + li * 0.022; d3 = t - tl
            if im is None or d3 < 0: continue
            u = ease_out(d3 / 0.18)
            paste(img, im, W / 2 + lx, 1105 + 16 * (1 - u), alpha=u * 0.92 * fo)
    return img.convert("RGB")

BURN_TXT_N = cv2.GaussianBlur(smooth_noise(H, W, 60, 21), (0, 0), 6) * 0.65 + 0.35 * (np.linspace(1, 0, W, dtype=np.float32)[None, :])
BURN_TXT_N = (BURN_TXT_N - BURN_TXT_N.min()) / (BURN_TXT_N.max() - BURN_TXT_N.min())

def render_bytes(i):
    return frame(i / FPS).tobytes()

if __name__ == "__main__":
    out = sys.argv[2]
    if len(sys.argv) > 3:
        spec = sys.argv[3]
        os.makedirs(out, exist_ok=True)
        if spec.startswith("seq:"):
            _, a, b = spec.split(":")
            times = [i / FPS for i in range(int(float(a) * FPS), int(float(b) * FPS))]
        else:
            times = [float(x) for x in spec.split(",")]
        from multiprocessing import Pool
        with Pool(4) as pool:
            ims = pool.map(frame, times)
        for tt, im in zip(times, ims):
            im.save(os.path.join(out, f"pv_{tt:06.3f}.jpg"), quality=88)
        sys.exit()
    from multiprocessing import Pool
    N = int(round(DUR * FPS))
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS), out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for i, b in enumerate(pool.imap(render_bytes, range(N), chunksize=3)):
            p.stdin.write(b)
            if i % 120 == 0: print(f"frame {i}/{N}", flush=True)
    p.stdin.close(); p.wait(); print("video done", p.returncode)
