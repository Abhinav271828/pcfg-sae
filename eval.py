from sae import SAEData, SAE
import torch
import os
import pickle as pkl
from model import GPT
from dgp import get_dataloader
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from evals import grammar_evals

# Load model and data
path = 'results/scratch/12owob2t'
state_dict = torch.load(os.path.join(path, 'latest_ckpt.pt'), map_location='cuda')
cfg = state_dict['config']

with open(os.path.join(path, 'grammar/PCFG.pkl'), 'rb') as f:
    pcfg = pkl.load(f)
model = GPT(cfg.model, pcfg.vocab_size).to('cuda')
model.load_state_dict(state_dict['net'])
model.eval()
dataloader = get_dataloader(
        language=cfg.data.language,
        config=cfg.data.config,
        alpha=cfg.data.alpha,
        prior_type=cfg.data.prior_type,
        num_iters=cfg.data.num_iters * cfg.data.batch_size,
        max_sample_length=cfg.data.max_sample_length,
        seed=cfg.seed,
        batch_size=cfg.data.batch_size,
        num_workers=0,
    )

# Load SAE
layer_name = 'wte'

def get_config(idx):
    return json.load(open(os.path.join(path, F'sae_{idx}/config.json')))

def get_sae(idx):
    config = get_config(idx)
    data = SAEData(model_dir=path, ckpt='latest_ckpt.pt', layer_name=config['layer_name'], device='cuda')
    embedding_size = data[0][0].size(-1)
    args = json.load(open(os.path.join(path, f'sae_{idx}/config.json')))
    sae = SAE(embedding_size, args['exp_factor'] * embedding_size, k=args['k'] if 'k' in args else None, sparsemax=args['sparsemax'] if 'sparsemax' in args else False).to('cuda')
    state_dict = torch.load(os.path.join(path, f'sae_{idx}/model.pth'), map_location='cpu')
    sae.load_state_dict(state_dict)
    sae.eval()
    return sae

# Evaluate intervened accuracy
validities = []
stds = []
for i in tqdm(range(494)):
    sae = get_sae(i)
    config = get_config(i)

    def hook(module, input, output):
        return sae(output)[1]

    if config['layer_name'] == 'wte':
        module = model.transformer.wte
    elif config['layer_name'] == "wpe":
        module = model.transformer.wpe
    elif config['layer_name'] == "attn0":
        module = model.transformer.h[0].attn
    elif config['layer_name'] == "mlp0":
        module = model.transformer.h[0].mlp
    elif config['layer_name'] == "res0":
        module = model.transformer.h[0]
    elif config['layer_name'] == "attn1":
        module = model.transformer.h[1].attn
    elif config['layer_name'] == "mlp1":
        module = model.transformer.h[1].mlp
    elif config['layer_name'] == "res1":
        module = model.transformer.h[1]
    elif config['layer_name'] == "ln_f":
        module = model.transformer.ln_f
    handle = module.register_forward_hook(hook)

    current = []
    for _ in range(5):
        results_after = grammar_evals(cfg, model, template=dataloader.dataset.template, grammar=dataloader.dataset.PCFG, device='cuda')
        current.append(results_after['validity'])

    validity = torch.tensor(current).mean().item()
    std = torch.tensor(current).std().item()
    with open('validities.txt', 'a') as f:
        f.write(f"{i} {validity} +- {std}\n")
    handle.remove()

