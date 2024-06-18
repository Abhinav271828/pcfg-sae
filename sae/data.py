import torch

from model import GPT
from dgp import get_dataloader

import os
import pickle as pkl
from tqdm import tqdm

path = 'results/scratch/m9z0jetl'

class SAEData:
    def __init__(self, data_path : str, model_dir : str, ckpt : str, layer_name : str, num_samples : int = -1):
        """
        A class to generate data to train the SAE. It extracts activations from a GPT model and saves them to a file.
        params:
            * data_path  : path to the data dir. If model_dir and ckpt are provided, the data is saved to this path;
                           otherwise, the data is loaded from this path
            * model_dir  : path to the directory containing the model
            * ckpt       : name of the checkpoint file
            * layer_name : name of the layer to extract activations from. one of
                - "wte" [embedding layer]
                - "wpe" [positional encoding layer]
                - (n, "attn") : n-th attention layer; n = 0, 1
                - (n, "mlp") : n-th mlp layer; n = 0, 1
                - "ln_f" [final layer-norm before the LM-head]
            * num_samples: number of samples to generate activations for. If -1, all samples are used.
        
        The saved data consists of two tensors:
            * sequences  : [num_samples * batch_size, seq_len] the input sequences
            * activations: [num_samples * batch_size, seq_len, emb_dim] the activations
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.ckpt = ckpt
        self.layer_name = layer_name
        self.num_samples = num_samples

        if self.model_dir and self.ckpt:
            model_dict = torch.load(os.path.join(self.model_dir, self.ckpt))
            cfg = model_dict['config']
            with open(os.path.join(self.model_dir, 'grammar/PCFG.pkl'), 'rb') as f:
                pcfg = pkl.load(f)
            self.model = GPT(cfg.model, pcfg.vocab_size)
            self.model.load_state_dict(model_dict['net'])
            self.model.eval()
            self.dataloader = get_dataloader(
                                language=cfg.data.language,
                                config=cfg.data.config,
                                alpha=cfg.data.alpha,
                                prior_type=cfg.data.prior_type,
                                num_iters=cfg.data.num_iters * cfg.data.batch_size,
                                max_sample_length=cfg.data.max_sample_length,
                                seed=cfg.seed,
                                batch_size=cfg.data.batch_size,
                                num_workers=cfg.data.num_workers)
            self.generate_activations()
            self.save_data()
        else:
            self.load_data()
    
    def save_data(self):
        save_dir = self.model_dir if self.model_dir else self.data_path
        torch.save(self.sequences, os.path.join(save_dir, 'sae_sequences.pt'))
        torch.save(self.activations, os.path.join(save_dir, 'sae_activations.pt'))

    def load_data(self):
        self.sequences = torch.load(os.path.join(self.data_path, 'sae_sequences.pt'))
        self.activations = torch.load(os.path.join(self.data_path, 'sae_activations.pt'))

    def generate_activations(self):
       
        activation = []
        match self.layer_name:
            case "wte":
                handle = self.model.transformer.wte.register_forward_hook(lambda model, input, output: activation.append(output.detach()))
            case "wpe":
                handle = self.model.transformer.wpe.register_forward_hook(lambda model, input, output: activation.append(output.detach()))
            case (n, "attn"):
                handle = self.model.transformer.h[n].attn.register_forward_hook(lambda model, input, output: activation.append(output.detach()))
            case (n, "mlp"):
                handle = self.model.transformer.h[n].mlp.register_forward_hook(lambda model, input, output: activation.append(output.detach()))
            case "ln_f":
                handle = self.model.transformer.ln_f.register_forward_hook(lambda model, input, output: activation.append(output.detach()))

        sequences = []
        i = 0
        for sequence, length in tqdm(self.dataloader, desc='Extracting', total=self.num_samples if self.num_samples > 0 else len(self.dataloader)):
            self.model(sequence)
            sequences.append(sequence)
            i += 1
            if self.num_samples > 0 and i >= self.num_samples:
                break

        handle.remove()

        self.activations = torch.cat(activation, dim=0)
        # [num_samples * batch_size, seq_len, emb_dim]
        self.sequences = torch.cat(sequences, dim=0)
        # [num_samples * batch_size, seq_len]

"""
GPT(
  (transformer): ModuleDict(
    (wte): Embedding(31, 128)
    (wpe): Embedding(256, 128)
    (h): ModuleList(
      (0-1): 2 x Block(
        (ln_1): LayerNorm()
        (attn): CausalSelfAttention(
          (c_attn): Linear(in_features=128, out_features=384, bias=False)
          (c_proj): Linear(in_features=128, out_features=128, bias=False)
        )
        (ln_2): LayerNorm()
        (mlp): MLP(
          (c_fc): Linear(in_features=128, out_features=512, bias=False)
          (gelu): GELU(approximate='none')
          (c_proj): Linear(in_features=512, out_features=128, bias=False)
        )
      )
    )
    (ln_f): LayerNorm()
  )
  (LM_head): Linear(in_features=128, out_features=31, bias=False)
)
"""