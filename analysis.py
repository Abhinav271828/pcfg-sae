import torch
from model import GPT
import os
import pickle as pkl

from sae import SAEData

path = 'results/scratch/t3m8xzkd'

state_dict = torch.load(os.path.join(path, 'latest_ckpt.pt'), map_location='cpu')
cfg = state_dict['config']

with open(os.path.join(path, 'grammar/PCFG.pkl'), 'rb') as f:
    pcfg = pkl.load(f)
model = GPT(cfg.model, pcfg.vocab_size)
model.load_state_dict(state_dict['net'])
model.eval()

data = SAEData(data_path=path, model_dir=path, ckpt='latest_ckpt.pt', layer_name='wte', num_samples=100, is_val=True, device='cpu').dataloader.dataset