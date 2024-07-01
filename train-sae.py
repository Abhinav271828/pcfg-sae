import argparse
from sae import train

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",    help="Path to the model directory",
                    type=str,         default=None,  required=True)
parser.add_argument("--ckpt",         help="Checkpoint file",
                    type=str,         default=None,  required=True)
parser.add_argument("--layer_name",   help="Layer name",
                    type=str,         default=None,  required=True)
parser.add_argument("--batch_size",   help="Batch size",
                    type=int,         default=128,   required=False)
parser.add_argument("--exp_factor",   help="Expansion factor",
                    type=int,         default=1,     required=False)
parser.add_argument("--alpha",        help="L1 regularization coefficient",
                    type=float,       default=None,  required=False)
parser.add_argument("--k",            help="Sparsity level",
                    type=int,         default=None,  required=False)
parser.add_argument("--sparsemax",    help="Use sparsemax instead of softmax",
                    type=bool,        default=False, required=False)
parser.add_argument("--lr",           help="Learning rate",
                    type=float,       default=1e-6,  required=False)
parser.add_argument("--train_iters",  help="Number of iterations to train",
                    type=int,         default=1000,  required=False)
parser.add_argument("--val_iters",    help="Number of iterations to validate",
                    type=int,         default=10,    required=False)
parser.add_argument("--val_interval", help="Validation interval",
                    type=int,         default=50,    required=False)
parser.add_argument("--patience",     help="Patience",
                    type=int,         default=3,     required=False)

args = parser.parse_args()
if sum([args.alpha is not None, args.k is not None, args.sparsemax]) != 1:
    raise ValueError("At most one of alpha, k, and sparsemax can be specified.")

train(args)