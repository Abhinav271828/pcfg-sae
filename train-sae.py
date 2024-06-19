import argparse
from sae import train

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

train(args)