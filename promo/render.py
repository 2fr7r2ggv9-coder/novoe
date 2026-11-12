"""Promo renderer: GUTROB — Шлёп, шлёп. 1080x1920, 30fps, 15s."""
import sys, math, subprocess, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

D = sys.argv[1]                      # work dir with hero.jpg, hero_cut.png, fonts/
OUT = sys.argv[2]
PREVIEW = len(sys.argv) > 3          # render only every Nth frame to PNG for checks
W, H, FPS, DUR = 1080, 1920, 30, 15.0
T0 = 105.45                          # song offset
rng = np.random.default_rng(7)

def L(t):  # global song time -> local
    return t - T0

BEATS = [L(b) for b in [105.19,105.79,106.39,107.0,107.6,108.2,108.81,109.41,110.02,110.62,111.25,
         111.85,112.45,113.06,113.66,114.26,114.87,115.47,116.08,116.68,117.28,117.91,118.51,
         119.12,119.72,120.33,120.93]]
SHLEP = [L(x) for x in [105.68, 106.24, 110.44, 111.08]]
OUTRO = L(115.47)  # title slam on the beat right after the last lyric

LINES = [  # (words with local start times), line end
    ([("ухожу",106.78),("без",107.10),("оглядки",107.36)], 108.05),
    ([("шлёпки",108.10),("стучат",108.58),("по",109.04),("пустой",109.20),("дорожке",109.70)], 110.40),
    ([("угас",111.48),("последний",111.68),("уголь",112.50)], 112.98),
    ([("а",113.02),("любовь",113.08),("сгорела,",113.40),("как",113.98),("сухой",114.08),("листок",114.58)], 115.40),
]
LINES = [([(w, L(s)) for w, s in ws], L(e)) for ws, e in LINES]

F_RUS = os.path.join(D, "fonts/RuslanDisplay.ttf")
F_ALE = os.path.join(D, "fonts/AlegreyaSC.ttf")
F_OSW = os.path.join(D, "fonts/Oswald.ttf")

# ---------- assets ----------
src = Image.open(os.path.join(D, "hero.jpg")).convert("RGB")
cut = Image.open(os.path.join(D, "hero_cut.png")).convert("RGBA")
SW, SH = src.size
# clean background plate: inpaint character area
a = np.array(cut)[:, :, 3]
mask = cv2.dilate((a > 20).astype(np.uint8) * 255, np.ones((25, 25), np.uint8))
plate = cv2.inpaint(np.array(src)[:, :, ::-1], mask, 21, cv2.INPAINT_TELEA)[:, :, ::-1]
plate = Image.fromarray(plate).filter(ImageFilter.GaussianBlur(1.2))
# soften cutout edge a bit
ca = Image.fromarray(a).filter(ImageFilter.GaussianBlur(1.0))
cut.putalpha(ca)

# extend canvas vertically (sky above, path below) for a wider vertical shot
TOPX, BOTX = 250, 126
def extend(im, mode):
    w, h = im.size
    out = Image.new(mode, (w, h + TOPX + BOTX), (0, 0, 0, 0) if mode == "RGBA" else None)
    if mode == "RGB":
        ref = np.asarray(im.crop((0, 0, 560, 50))).reshape(-1, 3)
        col = np.median(ref, axis=0)
        g = np.linspace(1.04, 1.0, TOPX + 60)[:, None, None]
        skyarr = np.clip(np.ones((TOPX + 60, w, 3)) * col[None, None, :] * g, 0, 255).astype(np.uint8)
        sky = Image.fromarray(skyarr)
        out.paste(sky, (0, 0))
        bot = im.crop((0, h - BOTX, w, h)).transpose(Image.FLIP_TOP_BOTTOM).filter(ImageFilter.GaussianBlur(6))
        out.paste(bot, (0, h + TOPX))
        # blend seams
        out.paste(im, (0, TOPX))
        arr = np.asarray(out).astype(np.float32)
        top = np.asarray(sky).astype(np.float32)
        for i in range(60):
            a = 1 - i / 60
            arr[TOPX + i] = arr[TOPX + i] * (1 - a) + top[TOPX + i] * a
        return Image.fromarray(arr.astype(np.uint8))
    out.paste(im, (0, TOPX))
    return out
plate = extend(plate, "RGB")
cut = extend(cut, "RGBA")
SW, SH = plate.size
FEET = (656.0, 975.0 + TOPX)
CX, CY = 610.0, SH / 2 + 10

# fog texture (tileable-ish), 2x width
def make_fog(w, h, seed, blur):
    r = np.random.default_rng(seed)
    n = r.random((h // 16, w // 16)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    n = cv2.GaussianBlur(n, (0, 0), blur)
    n = (n - n.min()) / (n.max() - n.min())
    return n
FOG_BACK = make_fog(2400, 1920, 1, 40)
FOG_FRONT = make_fog(2400, 1920, 2, 60)
yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
FOG_FRONT_MASK = np.clip((yy - 0.62) / 0.38, 0, 1) ** 1.5   # lower part
FOG_BACK_MASK = np.clip(1 - np.abs(yy - 0.55) / 0.3, 0, 1)   # horizon band

GRAIN = [rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]
VIGN = None
xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
yv = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
VIGN = np.clip(1 - 0.55 * ((xx ** 2) * 0.9 + (yv ** 2) * 0.6) ** 1.2, 0, 1)[:, :, None]

# embers
NE = 140
EMB = dict(x=rng.uniform(0, W, NE), y0=rng.uniform(0, H * 1.3, NE), sp=rng.uniform(60, 220, NE),
           sz=rng.uniform(2, 7, NE), ph=rng.uniform(0, 6.28, NE), wob=rng.uniform(10, 50, NE))
# ash/snow flakes (always)
NA = 90
ASH = dict(x=rng.uniform(0, W, NA), y0=rng.uniform(0, H, NA), sp=rng.uniform(25, 70, NA),
           sz=rng.uniform(1.5, 3.5, NA), ph=rng.uniform(0, 6.28, NA))

# ---------- text prerender ----------
def text_img(txt, font, size, fill=(240, 236, 228), stroke=0, stroke_fill=(20, 16, 12),
             shadow=True, glow=None, spacing=0):
    f = ImageFont.truetype(font, size)
    if spacing:
        widths = [f.getlength(c) for c in txt]
        tw = int(sum(widths) + spacing * (len(txt) - 1))
    else:
        tw = int(f.getlength(txt))
    asc, desc = f.getmetrics()
    pad = size // 2
    im = Image.new("RGBA", (tw + pad * 2, asc + desc + pad * 2), (0, 0, 0, 0))
    def draw(dr, col, off=(0, 0), sw=stroke, sf=stroke_fill):
        if spacing:
            x = pad
            for c, cw in zip(txt, widths):
                dr.text((x + off[0], pad + off[1]), c, font=f, fill=col, stroke_width=sw, stroke_fill=sf)
                x += cw + spacing
        else:
            dr.text((pad + off[0], pad + off[1]), txt, font=f, fill=col, stroke_width=sw, stroke_fill=sf)
    if shadow:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw(ImageDraw.Draw(sh), (0, 0, 0, 200), (0, size // 18), sw=stroke + 2, sf=(0, 0, 0, 200))
        sh = sh.filter(ImageFilter.GaussianBlur(size / 10))
        im.alpha_composite(sh)
    if glow:
        g = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw(ImageDraw.Draw(g), glow, sw=stroke + 4, sf=glow)
        g = g.filter(ImageFilter.GaussianBlur(size / 6))
        im.alpha_composite(g)
    draw(ImageDraw.Draw(im), fill)
    return im

STAMP = text_img("ШЛЁП", F_RUS, 225, fill=(245, 238, 225), stroke=3, glow=(255, 140, 40, 150))
TOP = text_img("GUTROB", F_RUS, 64, fill=(235, 230, 220), spacing=14)
TITLE1 = text_img("ШЛЁП,", F_RUS, 210, fill=(248, 240, 226), stroke=3, glow=(255, 130, 30, 170))
TITLE2 = text_img("ШЛЁП", F_RUS, 210, fill=(248, 240, 226), stroke=3, glow=(255, 130, 30, 170))
ART = text_img("GUTROB", F_RUS, 110, fill=(235, 230, 220), spacing=22)
CTA = text_img("СЛУШАЙ ПО ССЫЛКЕ В ШАПКЕ", F_OSW, 52, fill=(230, 224, 214), spacing=4)

WSIZE = 88
WCOL = (238, 232, 222)
WHOT = (255, 172, 70)
LINE_IMGS = []
for ws, end in LINES:
    words = []
    for w, s in ws:
        words.append((text_img(w.upper(), F_ALE, WSIZE, fill=WCOL, stroke=2),
                      text_img(w.upper(), F_ALE, WSIZE, fill=WHOT, stroke=2, glow=(255, 120, 20, 140)), s))
    LINE_IMGS.append((words, ws[0][1], end))

def layout_line(words, maxw=960):
    """returns rows of (index, x_offset) with word gap; wraps."""
    sp = 22
    pad = WSIZE // 2
    rows, cur, cw = [], [], 0
    for i, (im, _, _) in enumerate(words):
        ww = im.width - 2 * pad
        if cur and cw + sp + ww > maxw:
            rows.append((cur, cw)); cur, cw = [], 0
        cur.append((i, cw + (sp if cur else 0))); cw += (sp if cur[:-1] else 0) + ww
    rows.append((cur, cw))
    return rows
LINE_LAYOUT = [layout_line(w) for w, _, _ in LINE_IMGS]

def paste_center(canvas, im, cx, cy, scale=1.0, rot=0.0, alpha=1.0):
    if alpha <= 0.01:
        return
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BICUBIC)
    if rot:
        im = im.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 1:
        a = im.getchannel("A").point(lambda v: int(v * alpha))
        im = im.copy(); im.putalpha(a)
    canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

# ---------- motion helpers ----------
def last_event(t, evs):
    p = [e for e in evs if e <= t]
    return (t - p[-1], len(p) - 1) if p else (None, -1)

def ease_out(x):
    return 1 - (1 - min(max(x, 0), 1)) ** 3

def affine_for(scale, cx, cy, rot=0.0, ox=W / 2, oy=H / 2):
    # output->input mapping
    c, s = math.cos(rot), math.sin(rot)
    # x_in = cx + ( c*(x-ox) + s*(y-oy) )/scale
    a = c / scale; b = s / scale; d = -s / scale; e = c / scale
    return (a, b, cx - a * ox - b * oy, d, e, cy - d * ox - e * oy)

def compose_aff(outer, inner):
    # returns mapping x -> inner(outer(x)) where both are 6-tuples (out->in)
    a1, b1, c1, d1, e1, f1 = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (a2 * a1 + b2 * d1, a2 * b1 + b2 * e1, a2 * c1 + b2 * f1 + c2,
            d2 * a1 + e2 * d1, d2 * b1 + e2 * e1, d2 * c1 + e2 * f1 + f2)

# ---------- frame ----------
def frame(t):
    # camera
    S0 = H / SH
    push = 1.0 + 0.08 * (t / DUR)
    db, bi = last_event(t, BEATS)
    beat_p = 0.022 * math.exp(-db * 9) if db is not None else 0
    ds, si = last_event(t, SHLEP + [OUTRO])
    hit = math.exp(-ds * 7) if ds is not None and ds < 1.2 else 0
    scale = S0 * push * (1 + beat_p + 0.06 * hit)
    shake = hit * 10
    sx = shake * math.sin(t * 91) ; sy = shake * math.cos(t * 77)
    rot = math.radians(0.6 * hit * math.sin(t * 60))
    cx = CX + 6 * math.sin(t * 0.4) + sx / scale
    cy = CY - 30 * (t / DUR) + sy / scale
    cam = affine_for(scale, cx, cy, rot)

    bg = plate.transform((W, H), Image.AFFINE, cam, Image.BICUBIC)
    arr = np.asarray(bg).astype(np.float32) / 255.0

    # back fog band drifting
    off = int((t * 40) % 1200)
    fb = FOG_BACK[:, off:off + W] * FOG_BACK_MASK
    arr = arr * (1 - 0.35 * fb[:, :, None]) + 0.82 * 0.35 * fb[:, :, None]

    # character: walks toward camera, steps on beats
    k = 1.0 + 0.09 * ease_out(t / 11.0)
    step = 0.0
    side = 1
    if db is not None:
        ph = min(db / 0.604, 1.0)
        step = math.sin(ph * math.pi)          # up-down per step
        side = 1 if bi % 2 == 0 else -1
    bob = 7 * step + 10 * hit
    tilt = math.radians(1.1 * side * math.sin(min(db or 0, 0.6) / 0.6 * math.pi))
    fx, fy = FEET
    # inner: background-input coords -> cutout coords (scale k about feet, bob, tilt)
    c, s = math.cos(tilt), math.sin(tilt)
    a_ = c / k; b_ = s / k; d_ = -s / k; e_ = c / k
    inner = (a_, b_, fx - a_ * fx - b_ * (fy - bob) , d_, e_, fy - d_ * fx - e_ * (fy - bob))
    char = cut.transform((W, H), Image.AFFINE, compose_aff(cam, inner), Image.BICUBIC)
    ch = np.asarray(char).astype(np.float32) / 255.0
    al = ch[:, :, 3:4]
    # warm rim light pulse on hits
    crgb = ch[:, :, :3] * (1 + 0.12 * hit)
    arr = arr * (1 - al) + crgb * al

    # front fog
    off2 = int((t * 70) % 1200)
    ff = FOG_FRONT[:, off2:off2 + W] * FOG_FRONT_MASK
    arr = arr * (1 - 0.45 * ff[:, :, None]) + 0.85 * 0.45 * ff[:, :, None]

    # particles layer (half res)
    pl = Image.new("RGBA", (W // 2, H // 2), (0, 0, 0, 0))
    pd = ImageDraw.Draw(pl)
    for i in range(NA):
        y = (ASH["y0"][i] + ASH["sp"][i] * t) % H
        x = ASH["x"][i] + 18 * math.sin(t * 0.8 + ASH["ph"][i])
        r = ASH["sz"][i] / 2
        pd.ellipse((x / 2 - r, y / 2 - r, x / 2 + r, y / 2 + r), fill=(235, 235, 240, 130))
    ember_amt = 0.0
    if t > SHLEP[2] - 0.3:
        ember_amt = min(1, (t - (SHLEP[2] - 0.3)) / 1.5)
    if ember_amt > 0:
        n = int(NE * ember_amt)
        for i in range(n):
            y = H * 1.15 - ((EMB["y0"][i] + EMB["sp"][i] * (t + 2)) % (H * 1.3))
            x = EMB["x"][i] + EMB["wob"][i] * math.sin(t * 2 + EMB["ph"][i])
            fl = 0.6 + 0.4 * math.sin(t * 9 + EMB["ph"][i] * 3)
            r = EMB["sz"][i] / 2 * (1 + 0.4 * hit)
            pd.ellipse((x / 2 - r, y / 2 - r, x / 2 + r, y / 2 + r),
                       fill=(255, int(120 + 60 * fl), 40, int(230 * fl)))
    glow = pl.filter(ImageFilter.GaussianBlur(3))
    pl = Image.alpha_composite(glow, pl).resize((W, H), Image.BILINEAR)
    pa = np.asarray(pl).astype(np.float32) / 255.0
    arr = arr * (1 - pa[:, :, 3:4]) + pa[:, :, :3] * pa[:, :, 3:4]

    # grade: cold shadows, slight warm highlights, contrast
    lum = arr.mean(axis=2, keepdims=True)
    arr = lum + (arr - lum) * 0.82
    arr = np.clip((arr - 0.5) * 1.2 + 0.47, 0, 1)
    arr = arr + (1 - arr) * 0 + np.array([-0.02, 0.0, 0.035]) * (1 - lum) + np.array([0.03, 0.012, -0.01]) * lum
    # ember warmth at end
    arr = arr + np.array([0.05, 0.02, -0.02]) * ember_amt * 0.6
    arr *= VIGN
    # flash on hits
    if ds is not None and ds < 0.12:
        arr = arr + (1 - arr) * (0.55 * (1 - ds / 0.12))
    g = cv2.resize(GRAIN[int(t * FPS) % 6], (W, H), interpolation=cv2.INTER_NEAREST)
    arr = arr + g[:, :, None] * 0.028
    # outro dim
    if t > OUTRO:
        arr = arr * (1 - 0.35 * min(1, (t - OUTRO) / 0.4))
    # fade out
    if t > DUR - 0.7:
        arr = arr * max(0, (DUR - t) / 0.7)
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).convert("RGBA")

    # ---- text ----
    fade_out = max(0, (DUR - t) / 0.7) if t > DUR - 0.7 else 1
    if t < OUTRO:
        paste_center(img, TOP, W / 2, 230, alpha=0.85 * fade_out)
    # stamps
    for j, st in enumerate(SHLEP):
        dt = t - st
        if 0 <= dt < 0.65:
            sc = 1.0 + 0.6 * math.exp(-dt * 14)
            al = min(1, dt / 0.04) * (1 if dt < 0.4 else max(0, 1 - (dt - 0.4) / 0.25))
            x = W / 2 + (-60 if j % 2 == 0 else 60)
            y = 860 if j % 2 == 0 else 1040
            paste_center(img, STAMP, x, y, sc, rot=(6 if j % 2 == 0 else -6), alpha=al)
    # lyrics karaoke
    for li, (words, start, end) in enumerate(LINE_IMGS):
        if start - 0.15 <= t < end + 0.25:
            ain = ease_out((t - (start - 0.15)) / 0.25)
            aout = 1 if t < end else max(0, 1 - (t - end) / 0.25)
            rows = LINE_LAYOUT[li]
            base_y = 1290 - (len(rows) - 1) * 55
            for ri, (row, rw) in enumerate(rows):
                y = base_y + ri * 112
                x0 = W / 2 - rw / 2
                for wi, xo in row:
                    norm, hot, ws = words[wi]
                    wt = t - ws
                    pad = WSIZE // 2
                    cxw = x0 + xo + (norm.width - 2 * pad) / 2
                    if wt < 0:
                        paste_center(img, norm, cxw, y + 18 * (1 - ain), alpha=0.35 * ain * aout)
                    else:
                        pop = 1 + 0.18 * math.exp(-wt * 12)
                        lift = -10 * math.exp(-wt * 10)
                        im = hot if wt < 0.45 else norm
                        paste_center(img, im, cxw, y + lift, pop, alpha=aout)
    # outro title
    if t >= OUTRO:
        dt = t - OUTRO
        a1 = ease_out(dt / 0.15) * fade_out
        sc1 = 1 + 0.5 * math.exp(-dt * 12)
        paste_center(img, TITLE1, W / 2, 720, sc1, rot=-4, alpha=a1)
        dt2 = dt - 0.604
        if dt2 >= 0:
            sc2 = 1 + 0.5 * math.exp(-dt2 * 12)
            paste_center(img, TITLE2, W / 2, 930, sc2, rot=4, alpha=ease_out(dt2 / 0.15) * fade_out)
        dt3 = dt - 1.2
        if dt3 >= 0:
            # pulse with beats
            bp = 1 + 0.04 * (math.exp(-db * 9) if db is not None else 0)
            paste_center(img, ART, W / 2, 1150, bp, alpha=ease_out(dt3 / 0.4) * fade_out)
        dt4 = dt - 2.0
        if dt4 >= 0:
            paste_center(img, CTA, W / 2, 1290, alpha=ease_out(dt4 / 0.5) * fade_out * 0.9)
    return img.convert("RGB")

if PREVIEW:
    times = [float(x) for x in sys.argv[3].split(",")]
    for tt in times:
        frame(tt).save(os.path.join(OUT, f"pv_{tt:05.2f}.jpg"), quality=88)
    sys.exit()

N = int(DUR * FPS)
cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "-", "-ss", str(T0), "-t", str(DUR), "-i", os.path.join(D, "audio_st.wav"),
       "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
       "-af", f"afade=t=in:d=0.03,afade=t=out:st={DUR-0.8}:d=0.8", "-c:a", "aac", "-b:a", "256k",
       "-movflags", "+faststart", "-shortest", OUT]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i in range(N):
    p.stdin.write(frame(i / FPS).tobytes())
    if i % 60 == 0:
        print(f"frame {i}/{N}", flush=True)
p.stdin.close(); p.wait()
print("done", p.returncode)
