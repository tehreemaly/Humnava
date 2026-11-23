import sys, io, yaml, torch, numpy as np
from argparse import Namespace
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File
import easyocr

sys.path.insert(0, 'EasyOCR/trainer')
from utils import CTCLabelConverter
from model import Model

MODEL_PATH = 'models/humnava_ocr_v2_ed.pth'   # change later to the best model

cfg = yaml.safe_load(open('EasyOCR/trainer/config_files/urdu_config.yaml', encoding='utf-8'))
conv = CTCLabelConverter(cfg['lang_char'])
opt = Namespace(**cfg, num_class=len(conv.character))

model = torch.nn.DataParallel(Model(opt))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'), strict=True)
model = model.cuda().eval()

detector = easyocr.Reader(['ur'], recognizer=False, gpu=True)

def prep(im):
    im = ImageOps.autocontrast(im.convert('L'), cutoff=2)
    w, h = im.size
    nw = min(600, max(1, round(64 * w / h)))
    a = (np.asarray(im.resize((nw, 64), Image.BICUBIC), dtype=np.float32) / 255. - 0.5) / 0.5
    out = np.repeat(a[:, -1:], 600, axis=1)
    out[:, :nw] = a
    return torch.from_numpy(out)[None, None].cuda()

def decode(idx):
    s, prev = [], 0
    for i in idx:
        if i != 0 and i != prev:
            s.append(conv.character[i])
        prev = i
    return ''.join(s)[::-1]

def find_lines(img_np):
    boxes, _ = detector.detect(img_np, text_threshold=0.5, low_text=0.3, link_threshold=0.3, width_ths=2.0, min_size=10)
    boxes = boxes[0]                      # [x_min, x_max, y_min, y_max]
    boxes = sorted(boxes, key=lambda b: (b[2] + b[3]) / 2)
    lines = []
    for b in boxes:
        yc, h = (b[2] + b[3]) / 2, b[3] - b[2]
        if lines and abs(yc - lines[-1]['yc']) < 0.6 * lines[-1]['h']:
            L = lines[-1]
            L['x0'], L['x1'] = min(L['x0'], b[0]), max(L['x1'], b[1])
            L['y0'], L['y1'] = min(L['y0'], b[2]), max(L['y1'], b[3])
            L['yc'], L['h'] = (L['y0'] + L['y1']) / 2, L['y1'] - L['y0']
        else:
            lines.append(dict(x0=b[0], x1=b[1], y0=b[2], y1=b[3], yc=yc, h=h))
    return lines

app = FastAPI()

@app.post('/read')
async def read(file: UploadFile = File(...)):
    im = Image.open(io.BytesIO(await file.read())).convert('RGB')
    lines = find_lines(np.asarray(im))
    gx0 = min(L['x0'] for L in lines)
    gx1 = max(L['x1'] for L in lines)
    texts = []
    with torch.no_grad():
        for L in lines:
            crop = im.crop((max(0, gx0 - 25), max(0, L['y0'] - 4),
                            min(im.width, gx1 + 25), L['y1'] + 4))
            texts.append(decode(model(prep(crop), None).argmax(2)[0].tolist()))
    return {'lines': texts, 'text': '\n'.join(texts)}
