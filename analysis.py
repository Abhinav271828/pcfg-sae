import torch
from model import GPT
import os
import pickle as pkl

path = 'results/scratch/m9z0jetl'

state_dict = torch.load(os.path.join(path, 'latest_ckpt.pt'))
cfg = state_dict['config']

with open(os.path.join(path, 'grammar/PCFG.pkl'), 'rb') as f:
    pcfg = pkl.load(f)
model = GPT(cfg.model, pcfg.vocab_size)
model.load_state_dict(state_dict['net'])
model.eval()
