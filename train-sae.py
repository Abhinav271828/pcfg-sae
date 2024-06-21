import argparse
from sae import train

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",    help="Path to the model directory",
                    type=str,         default=None, required=True)
parser.add_argument("--ckpt",         help="Checkpoint file",
                    type=str,         default=None, required=True)
parser.add_argument("--layer_name",   help="Layer name",
                    type=str,         default=None, required=True)
parser.add_argument("--batch_size",   help="Batch size",
                    type=int,         default=4,    required=False)
parser.add_argument("--exp_factor",   help="Expansion factor",
                    type=int,         default=1,    required=False)
parser.add_argument("--alpha",        help="L1 regularization coefficient",
                    type=float,       default=None, required=False)
parser.add_argument("--k",            help="Sparsity level",
                    type=int,         default=-1,   required=False)
parser.add_argument("--lr",           help="Learning rate",
                    type=float,       default=1e-6, required=False)
parser.add_argument("--train_iters",  help="Number of iterations to train",
                    type=int,         default=1000, required=False)
parser.add_argument("--val_iters",    help="Number of iterations to validate",
                    type=int,         default=10,   required=False)
parser.add_argument("--val_interval", help="Validation interval",
                    type=int,         default=50,   required=False)
parser.add_argument("--patience",     help="Patience",
                    type=int,         default=3,    required=False)

args = parser.parse_args()
if args.alpha is None and args.k == -1:
    raise ValueError("Exactly one of alpha or k must be specified.")

train(args)