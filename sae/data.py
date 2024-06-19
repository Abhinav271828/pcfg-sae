import torch
from torch.utils.data import Dataset

from model import GPT
from dgp import get_dataloader

import os
import pickle as pkl
from tqdm import tqdm

class SAEData(Dataset):
    def __init__(self, data_path : str, model_dir : str, ckpt : str, layer_name : str, num_samples : int = -1, device : str = 'cuda'):
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
            * activations: [total_tokens, emb_dim] the activations (flattened across batch dimension and padding tokens removed)
            * seq_ids    : [total_tokens] the sequence ids
        The data is valid if:
            * activations.size(0) == seq_ids.size(0) <= sequences.size(0)
                - [equality means there are no padding tokens in the sequences]
            * seq_ids[-1] = sequences.size(0) - 1
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.ckpt = ckpt
        self.layer_name = layer_name
        self.num_samples = num_samples
        self.device = device

        if self.model_dir and self.ckpt:
            print("Loading model...")
            model_dict = torch.load(os.path.join(self.model_dir, self.ckpt))
            cfg = model_dict['config']
            with open(os.path.join(self.model_dir, 'grammar/PCFG.pkl'), 'rb') as f:
                pcfg = pkl.load(f)
            self.model = GPT(cfg.model, pcfg.vocab_size).to(self.device)
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
            print("Generating activations...")
            self.generate_activations()
            self.save_data()
        else:
            print("Loading data...")
            self.load_data()
    
    def save_data(self):
        save_dir = self.model_dir if self.model_dir else self.data_path
        torch.save(self.sequences, os.path.join(save_dir, 'sae_sequences.pt'))
        torch.save(self.activations, os.path.join(save_dir, 'sae_activations.pt'))
        torch.save(self.seq_ids, os.path.join(save_dir, 'sae_seq_ids.pt'))

    def load_data(self):
        self.sequences = torch.load(os.path.join(self.data_path, 'sae_sequences.pt'))
        self.activations = torch.load(os.path.join(self.data_path, 'sae_activations.pt'))
        self.seq_ids = torch.load(os.path.join(self.data_path, 'sae_seq_ids.pt'))

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
        seq_ids = []
        i = 0
        seq = 0
        for sequence, length in tqdm(self.dataloader, desc='Extracting', total=self.num_samples if self.num_samples > 0 else len(self.dataloader)):
            self.model(sequence.to(self.device))
            length = [int(l) for l in length.tolist()]
            activation[-1] = torch.cat([activation[-1][i][:l] for i, l in enumerate(length)], dim=0)
            sequences.append(sequence)
            for l in length:
                seq_ids += [seq] * int(l)
                seq += 1
            i += 1
            if self.num_samples > 0 and i >= self.num_samples:
                break

        handle.remove()

        self.activations = torch.cat(activation, dim=0)
        # [total_tokens, emb_dim]
        self.sequences = torch.cat(sequences, dim=0)
        # [num_samples * batch_size, seq_len]
        self.seq_ids = torch.tensor(seq_ids)
        # [total_tokens]

    def __len__(self):
        return self.activations.size(0)

    def __getitem__(self, idx):
        return self.seq_ids[idx], self.activations[idx]

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

