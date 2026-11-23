import sys, os, yaml, torch, jiwer, numpy as np
from argparse import Namespace
from PIL import Image
from PIL import ImageOps
sys.path.insert(0, 'EasyOCR/trainer')
from utils import CTCLabelConverter
from model import Model

img_dir, tsv = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open('EasyOCR/trainer/config_files/urdu_config.yaml', encoding='utf-8'))
conv = CTCLabelConverter(cfg['lang_char'])
opt = Namespace(**cfg, num_class=len(conv.character))

def load(path):
    m = torch.nn.DataParallel(Model(opt))
    m.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
    return m.cuda().eval()

def prep(p):
    im = ImageOps.autocontrast(Image.open(p).convert('L'), cutoff=2)
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
    return ''.join(s)[::-1]  # back to normal reading order

rows = [l.rstrip('\n').split('\t', 1) for l in open(tsv, encoding='utf-8') if '\t' in l]
models = {'original': cfg['saved_model'], 'humnava_v1': 'models/humnava_ocr_v1.pth', 'v2_acc': 'models/humnava_ocr_v2_acc.pth', 'v2_ed': 'models/humnava_ocr_v2_ed.pth', 'v3_acc': 'models/humnava_ocr_v3_acc.pth', 'v3_ed': 'models/humnava_ocr_v3_ed.pth'}
for name, path in models.items():
    m = load(path)
    gts, preds = [], []
    with torch.no_grad():
        for fn, gt in rows:
            preds.append(decode(m(prep(os.path.join(img_dir, fn)), None).argmax(2)[0].tolist()))
            gts.append(gt.strip())
    exact = sum(g == p for g, p in zip(gts, preds)) / len(gts)
    print(f'{name}: CER={jiwer.cer(gts, preds):.3f}  exact={exact:.1%}  lines={len(gts)}')
    for g, p in list(zip(gts, preds))[:10]:
        print('GT  :', g); print('PRED:', p)
    if name == 'v3_ed':
        for g, p in zip(gts, preds):
            if g != p:
                print(f'ERR cer={jiwer.cer(g, p):.2f}\nERR GT  : {g}\nERR PRED: {p}\n')
    if name == 'v3_ed':
        ns = lambda t: [x.replace(' ', '') for x in t]
        print(f'NOSPACE CER={jiwer.cer(ns(gts), ns(preds)):.3f}')
    if name == 'v3_ed':
        import difflib, collections
        c = collections.Counter()
        for g, p in zip(gts, preds):
            g, p = g.replace(' ', ''), p.replace(' ', '')
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, g, p).get_opcodes():
                if tag == 'replace' and i2 - i1 == 1 and j2 - j1 == 1:
                    c[(g[i1], p[j1])] += 1
        for (a, b), n in c.most_common(15):
            print(f'CONF {a} -> {b} : {n}')
