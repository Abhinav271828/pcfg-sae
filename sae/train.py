from torch.utils.data import DataLoader

from .model import *
from .data import *

import wandb
from tqdm import tqdm
import json

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data = SAEData(args.model_dir, args.ckpt, args.layer_name, device=device)
    val_data = SAEData(args.model_dir, args.ckpt, args.layer_name, device=device)
    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    val_dl = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=val_data.collate_fn)

    embedding_size = train_data[0][0].size(-1)
    model = SAE(embedding_size, args.exp_factor * embedding_size, k=args.k, sparsemax=args.sparsemax).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Train the SAE
    wandb.init(project="sae")
    wandb.run.name = wandb.run.id
    wandb.run.save()
    wandb.config.update(vars(args))

    for epoch in range(1):
        loss_increasing = 0

        train_loss = 0
        prev_loss = float('inf')

        train_it = 0
        for activation, seq in tqdm(train_dl, desc="Training", total=args.train_iters):
            if train_it > args.train_iters: break

            activation = activation.to(device)
            optimizer.zero_grad()
            latent, recon = model(activation)
            recon_loss = criterion(recon, activation)
            reg_loss = args.alpha * torch.norm(latent, p=1) if args.alpha else 0
            loss = recon_loss + reg_loss
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            if loss > prev_loss: loss_increasing += 1
            else: loss_increasing = 0

            if loss_increasing == args.patience: break
            prev_loss = loss

            if train_it % args.val_interval == 0:
                model.eval()
                val_loss = 0
                val_it = 0
                with torch.no_grad():
                    for activation, seq in val_dl:
                        if val_it > args.val_iters: break
                        activation = activation.to(device)
                        latent, recon = model(activation)
                        loss = criterion(recon, activation)
                        val_loss += loss.item()
                        val_it += 1
                model.train()
                wandb.log({'recon_loss': recon_loss.item(),
                           'reg_loss'  : reg_loss.item() if args.alpha else 0,
                           'train_loss': train_loss,
                           'val_loss'  : val_loss   / args.val_iters})
            else:
                wandb.log({'recon_loss': recon_loss.item(),
                           'reg_loss'  : reg_loss.item() if args.alpha else 0,
                           'train_loss': train_loss})
            train_it += 1

    i = 0
    while True:
        dir_name = os.path.join(args.model_dir, f'sae_{i}')
        if not os.path.exists(dir_name): break
        i += 1
    os.mkdir(dir_name)
    torch.save(model.state_dict(), os.path.join(dir_name, 'model.pth'))
    with open(os.path.join(dir_name, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    wandb.finish()
