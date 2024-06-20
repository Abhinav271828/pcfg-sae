# Broad notes on usage

- The data-generating process is in folder *dgp*
- To run, simply use `python train.py` 
- The script uses Hydra and the train config can be located in folder *config*
- Some basic evals to check grammaticality of generations have been implemented in *evals*
- There is currently only free generation task in this codebase. To add more, check out places with *NOTE* in the code.

# Modifications
- `CORR` means that I tried to debug something
- We pass a language name and a config dict instead of a set of hyperparameters.
    - Language is one of `['english', 'expr', 'dyck']`
    - Hyperparams and grammar for `english` are unchanged
    - Hyperparams for `expr` are `n_digits` (number of digits), `n_ops` (number of operations), and `bracket` (whether or not there are brackets)
    - Hyperparam for `dyck` is `n_brackets` (number of different types of brackets)
- Default values for all are encoded in `config/train/conf.yaml`. Note that `config/debug` doesn't reflect any of the changes.
- Documentation modified accordingly.

# SAEs
- All SAE-related code is in `sae/`, except the argument parsing and top-level function call, which are in `train-sae.py`.
- Training the SAE makes use of the following hyperparameters, treated as command-line arguments to `train-sae.py`. For default values, please refer to the code.
    - `data_path`: The path to the folder that contains/should contain the SAE training data (model activations). For convenience, I keep it the same as the model dir itself, e.g. `results/scratch/t3m8xzkd`.
    - model_dir: If the SAE data is not already present in `data_path`, the path to the directory in which the GPT model that it is to be extracted from is in.
    - ckpt: If the SAE data is not already present in `data_path`, the filename of the GPT model that it is to be extracted from.
    - layer_name: The layer of the model whose outputs we want to disentangle. Options are:
        - `wte` [embedding layer]
        - `wpe` [positional encoding layer]
        - `attn{n}` [n-th attention layer; n = 0, 1]
        - `mlp{n}` [n-th mlp layer; n = 0, 1]
        - `ln_f` [final layer-norm before the LM-head]
    - num_samples: The number of batches taken from the input dataloader to generate activations.
    - batch_size: The batch size of the dataloader used to train the SAE.
    - exp_factor: The expansion factor of the SAE (the ratio of the hidden size to the input size).
    - alpha: If we want to use L1-regularization, the coefficient of the L1 norm of the latent in the loss function.
    - k: If we want to use top-k-regularization, the number of latent dimensions that are kept nonzero.
    - lr: The learning rate of the SAE. Used in an Adam optimizer.
    - num_epochs: The maximum number of epochs to train the SAE. Note that these many epochs may not be run if patience (below) is positive.
    - patience: If loss increases (not stays the same) for `patience` consecutive epochs, training is stopped.
- All of the above are optional arguments except `data_path` and `layer_name`; furthermore, one of `alpha` and `k` must be provided, although this is not enforced.
- The dataloader and GPT model are loaded according to the saved config file in the `model_dir`.
- For each model-layer combination, six files are saved in the model's directory; the following three for `split = train, val`:
    - the activations `{layer}_{split}_activations.pt`, containing the actual inputs to the autoencoder.
    - the sequence IDs `{layer}_{split}_seq_ids.pt`, containing the index of the sequence to which the token that generated the corresponding embedding is from.
    - the sequences `{layer}_{split}_sequences.pt`, containing the tokens that generated the embeddings, grouped into their sequences.
- Trained SAEs are saved in a subdirectory `sae_{i}` (depending on how many SAEs have been saved previously for the same model), which contains a `config.json` with the above arguments and a `model.pth` file.

# Model Checkpoints
- `results/scratch/t3m8xzkd`: Model trained on English.