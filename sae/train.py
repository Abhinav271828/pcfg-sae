from torch.utils.data import DataLoader

from .model import *
from .data import *

import argparse
import wandb
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",   help="Path to the data directory",
                    type=str)
parser.add_argument("--model_dir",   help="Path to the model directory",
                    type=str,        default=None, required=False)
parser.add_argument("--ckpt",        help="Checkpoint file",
                    type=str,        default=None, required=False)
parser.add_argument("--layer_name",  help="Layer name",
                    type=str,        default=None, required=False)
parser.add_argument("--num_samples", help="Number of samples",
                    type=int,        default=1024, required=False)
parser.add_argument("--batch_size",  help="Batch size",
                    type=int,        default=32,   required=False)
parser.add_argument("--exp_factor",  help="Expansion factor",
                    type=int,        default=1,    required=False)
parser.add_argument("--alpha",       help="L1 regularization coefficient",
                    type=float,      default=None, required=False)
parser.add_argument("--k",           help="Sparsity level",
                    type=int,        default=-1,   required=False)
parser.add_argument("--lr",          help="Learning rate",
                    type=float,      default=1e-6, required=False)
parser.add_argument("--num_epochs",  help="Number of epochs",
                    type=int,        default=100,  required=False)
parser.add_argument("--patience",    help="Patience",
                    type=int,        default=3,    required=False)

args = parser.parse_args()

train_data = SAEData(args.data_path, args.model_dir, args.ckpt, args.layer_name, args.num_samples)
dev_data = SAEData(args.data_path, args.model_dir, args.ckpt, args.layer_name, int(0.05 * args.num_samples))
train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_dl = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)

embedding_size = train_data.activations.size(-1)
model = SAE(embedding_size, args.exp_factor * embedding_size, k=-1 if args.alpha else args.k)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train the SAE
wandb.init(project="sae")
wandb.run.name = wandb.run.id
wandb.run.save()
wandb.config.update(vars(args))

loss_increasing = 0
prev_loss = float('inf')
for epoch in range(args.num_epochs):
    model.train()
    for seq_id, activation in tqdm(train_dl, desc="Training"):
        activation = activation.to(device)
        optimizer.zero_grad()
        latent, recon = model(activation)
        loss = criterion(recon, activation)
        loss += args.alpha * torch.norm(latent, p=1) if args.alpha else 0
        loss.backward()
        optimizer.step()

    wandb.log({'train_loss': loss.item()})
    print("Epoch: %d, Loss: %.4f" % (epoch, loss.item()))

    if loss > prev_loss: loss_increasing += 1
    else: loss_increasing = 0

    if loss_increasing == args.patience: break
    prev_loss = loss

    model.eval()
    with torch.no_grad():
        for seq_id, activation in val_dl:
            activation = activation.to(device)
            latent, recon = model(activation)
            loss = criterion(recon, activation)
        wandb.log({'val_loss': loss.item()})

i = 0
while True:
    dir_name = os.path.join(args.model_dir, f'sae_{i}')
    if not os.path.exists(dir_name): break
torch.save(model.state_dict(), os.path.join(dir_name, 'model.pth'))
with open(os.path.join(dir_name, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)
wandb.finish()