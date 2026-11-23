import os, sys, yaml, torch
from argparse import Namespace
from easyocr.config import recognition_models as R

chars = ''
for gen in R:
    if 'arabic_g1' in R[gen]:
        chars = R[gen]['arabic_g1']['characters']

rows = [l.split('\t', 1)[1].strip() for l in open('nastaliq_data/labels.tsv', encoding='utf-8') if '\t' in l]
max_len = max(len(r) for r in rows)
print('longest label:', max_len)

cfg = dict(
    number='', symbol='', lang_char=chars,
    experiment_name='urdu_ft', train_data='all_data', valid_data='all_data/urdu_val',
    manualSeed=1111, workers=4, batch_size=32, num_iter=4000, valInterval=500,
    saved_model=os.path.expanduser('~/.EasyOCR/model/arabic.pth'), FT=True,
    optim=False, lr=1.0, beta1=0.9, rho=0.95, eps=1e-8, grad_clip=5,
    select_data='urdu_train', batch_ratio='1', total_data_usage_ratio=1.0,
    batch_max_length=max_len + 10, imgH=64, imgW=600, rgb=False,
    contrast_adjust=0.0, sensitive=True, PAD=True, data_filtering_off=False,
    Transformation='None', FeatureExtraction='ResNet', SequenceModeling='BiLSTM',
    Prediction='CTC', num_fiducial=20, input_channel=1, output_channel=512,
    hidden_size=512, decode='greedy', new_prediction=False,
    freeze_FeatureFxtraction=False, freeze_SequenceModeling=False,
)
path = 'EasyOCR/trainer/config_files/urdu_config.yaml'
yaml.safe_dump(cfg, open(path, 'w', encoding='utf-8'), allow_unicode=True, sort_keys=False)
print('config saved:', path)

# check: do the pretrained weights fit this model exactly?
sys.path.insert(0, 'EasyOCR/trainer')
from utils import CTCLabelConverter
from model import Model
conv = CTCLabelConverter(chars)
opt = Namespace(**cfg, num_class=len(conv.character))
m = torch.nn.DataParallel(Model(opt))
m.load_state_dict(torch.load(cfg['saved_model'], map_location='cpu'), strict=True)
print('Pretrained weights load OK (strict). Classes:', len(conv.character))
