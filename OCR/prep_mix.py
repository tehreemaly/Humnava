"""
prep_mix.py - prepares mix_data for EasyOCR training (v1 -> v2)

Does two things:
1. Copies mix_data into EasyOCR/trainer/all_data_mix (mix_train + mix_val),
   with REVERSED labels (like prep_data.py)
2. Writes EasyOCR/trainer/config_files/urdu_mix_config.yaml
   (starts from models/humnava_ocr_v1.pth, lower learning rate)

Run:  cd ~/humnava && python prep_mix.py
"""
import os
import shutil
import sys
from argparse import Namespace

import torch
import yaml
from easyocr.config import recognition_models as R

SRC = 'mix_data'
DST = 'EasyOCR/trainer/all_data_mix'
START = os.path.expanduser('~/humnava/models/humnava_ocr_v1.pth')

chars = ''
for gen in R:
    if 'arabic_g1' in R[gen]:
        chars = R[gen]['arabic_g1']['characters']


def read(path):
    rows = []
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if '\t' not in line:
            continue
        fn, text = line.split('\t', 1)
        if os.path.exists(f'{SRC}/images/{fn}'):
            rows.append((fn, text.strip()))
    return rows


splits = {
    'mix_train': read(f'{SRC}/labels.tsv'),
    'mix_val': read(f'{SRC}/val.tsv'),
}

if os.path.exists(DST):
    shutil.rmtree(DST)

unknown = set()
max_len = 0
for name, items in splits.items():
    os.makedirs(f'{DST}/{name}')
    with open(f'{DST}/{name}/labels.csv', 'w', encoding='utf-8', newline='') as f:
        f.write('filename,words\n')
        for fn, text in items:
            shutil.copy(f'{SRC}/images/{fn}', f'{DST}/{name}/{fn}')
            f.write(f'{fn},{text[::-1]}\n')
            unknown.update(c for c in text if c not in chars)
            max_len = max(max_len, len(text))

print({k: len(v) for k, v in splits.items()}, 'longest label:', max_len)
if unknown:
    print('WARNING: these characters are not in the model character set:',
          ' '.join(sorted(unknown)))
    print('Fix them in train.tsv, or the model cannot learn them.')

cfg = dict(
    number='', symbol='', lang_char=chars,
    experiment_name='urdu_mix3', train_data='all_data_mix',
    valid_data='all_data_mix/mix_val',
    manualSeed=1111, workers=4, batch_size=16, num_iter=3000, valInterval=200,
    saved_model=START, FT=True,
    optim=False, lr=0.1, beta1=0.9, rho=0.95, eps=1e-8, grad_clip=5,
    select_data='mix_train', batch_ratio='1', total_data_usage_ratio=1.0,
    batch_max_length=max_len + 10, imgH=64, imgW=600, rgb=False,
    contrast_adjust=0.0, sensitive=True, PAD=True, data_filtering_off=False,
    Transformation='None', FeatureExtraction='ResNet', SequenceModeling='BiLSTM',
    Prediction='CTC', num_fiducial=20, input_channel=1, output_channel=512,
    hidden_size=512, decode='greedy', new_prediction=False,
    freeze_FeatureFxtraction=False, freeze_SequenceModeling=False,
)
path = 'EasyOCR/trainer/config_files/urdu_mix_config.yaml'
yaml.safe_dump(cfg, open(path, 'w', encoding='utf-8'),
               allow_unicode=True, sort_keys=False)
print('config saved:', path)

# check that v1 weights fit this model
sys.path.insert(0, 'EasyOCR/trainer')
from utils import CTCLabelConverter
from model import Model

conv = CTCLabelConverter(chars)
opt = Namespace(**cfg, num_class=len(conv.character))
m = torch.nn.DataParallel(Model(opt))
m.load_state_dict(torch.load(START, map_location='cpu'), strict=True)
print('v1 weights load OK. Classes:', len(conv.character))
