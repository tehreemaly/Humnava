import os, sys, yaml
os.chdir('EasyOCR/trainer')
sys.path.insert(0, os.getcwd())
from train import train

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

with open('config_files/urdu_mix_config.yaml', encoding='utf-8') as f:
    opt = AttrDict(yaml.safe_load(f))
opt.character = opt.lang_char
os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
train(opt, amp=False)
