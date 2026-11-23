
import sys, json, numpy as np, cv2
import onnxruntime as ort
from PIL import Image, ImageOps
from scipy.signal import find_peaks

chars = json.load(open('models/characters.json', encoding='utf-8'))
sess = ort.InferenceSession('models/humnava_ocr_int8.onnx')

def prep(im):
    im = ImageOps.autocontrast(im.convert('L'), cutoff=2)
    w, h = im.size
    nw = min(600, max(1, round(64 * w / h)))
    a = (np.asarray(im.resize((nw, 64), Image.BICUBIC), dtype=np.float32) / 255. - 0.5) / 0.5
    out = np.repeat(a[:, -1:], 600, axis=1)
    out[:, :nw] = a
    return out[None, None].astype(np.float32)

def decode(idx):
    s, prev = [], 0
    for i in idx:
        if i != 0 and i != prev:
            s.append(chars[i])
        prev = i
    return ''.join(s)[::-1]

def binarize(img, W=1000):
    s = W / img.width
    g = np.asarray(img.convert('L').resize((W, round(img.height * s))))
    g = cv2.GaussianBlur(g, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 41, 12)
    return bw, s

def best_angle(bw):
    best, ba = -1, 0
    h, w = bw.shape
    for a in np.arange(-6, 6.01, 0.25):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1)
        r = cv2.warpAffine(bw, M, (w, h))
        v = np.var(r.sum(1).astype(float))
        if v > best:
            best, ba = v, a
    return ba

def find_lines(img):
    bw, s = binarize(img)
    h, w = bw.shape
    ang = best_angle(bw)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
    bwr = cv2.warpAffine(bw, M, (w, h))
    rot = img.rotate(ang, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    prof = np.convolve(bwr.sum(1).astype(float), np.ones(5) / 5, mode='same')
    p = prof - prof.mean()
    ac = np.correlate(p, p, 'full')[len(p) - 1:]
    pitch = 15 + int(np.argmax(ac[15:120]))
    ref = np.percentile(prof, 90)
    ink_rows = np.where(prof > 0.04 * ref)[0]
    mins, _ = find_peaks(-prof, distance=int(pitch * 0.6))
    cuts = [c for c in mins if ink_rows[0] <= c <= ink_rows[-1]]
    cuts = sorted(set([ink_rows[0]] + cuts + [ink_rows[-1]]))
    out = []
    for y0, y1 in zip(cuts[:-1], cuts[1:]):
        if y1 - y0 < 0.5 * pitch:
            continue
        band = bwr[y0:y1]
        if band.sum() < 0.04 * ref * (y1 - y0) / 3:
            continue
        col = band.sum(0)
        xs = np.where(col >= 2)[0]
        if len(xs) == 0:
            continue
        groups, g0, prev = [], xs[0], xs[0]
        for x in xs[1:]:
            if x - prev > 60:
                groups.append((g0, prev)); g0 = x
            prev = x
        groups.append((g0, prev))
        best = max(groups, key=lambda t: col[t[0]:t[1] + 1].sum())
        out.append((int(best[0] / s), int(y0 / s), int(best[1] / s), int(y1 / s)))
    return rot, out


def split_line(crop, max_ar=8.0):
    w, h = crop.size
    ar = w / h
    if ar <= max_ar * 1.15:
        return [crop]
    k = int(np.ceil(ar / max_ar))
    g = np.asarray(crop.convert('L'))
    _, bw = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col = np.convolve(bw.sum(0).astype(float), np.ones(5) / 5, mode='same')
    cuts = [0]
    for i in range(1, k):
        t = int(i * w / k)
        r = int(0.10 * w / k)
        lo, hi = max(cuts[-1] + 10, t - r), min(w - 10, t + r)
        cuts.append(lo + int(np.argmin(col[lo:hi])))
    cuts.append(w)
    return [crop.crop((a, 0, b, h)) for a, b in zip(cuts[:-1], cuts[1:])]


def is_junk(t):
    toks = t.split()
    if not toks:
        return True
    letters = sum(c.isalpha() for c in t)
    singles = sum(len(w) == 1 for w in toks)
    return letters < 4 or singles / len(toks) > 0.5

img = ImageOps.exif_transpose(Image.open(sys.argv[1])).convert('RGB')
rot, lines = find_lines(img)
dbg = np.asarray(rot).copy()
texts = []
px = max(25, int(0.025 * rot.width))
for x0, y0, x1, y1 in lines:
    py = max(3, int(0.08 * (y1 - y0)))
    crop = rot.crop((max(0, x0 - px), max(0, y0 - py),
                     min(rot.width, x1 + px), min(rot.height, y1 + py)))
    parts = []
    for p in split_line(crop):
        lg = sess.run(None, {'image': prep(p)})[0]
        parts.append(decode(lg.argmax(2)[0].tolist()))
    t = ' '.join(reversed(parts))
    if not is_junk(t):
        texts.append(t)
    cv2.rectangle(dbg, (x0, y0), (x1, y1), (255, 0, 0), 3)
cv2.imwrite('lines_debug.jpg', cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
print(f'{len(lines)} lines found')
print('\n'.join(texts))

