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
    - model_dir: The path to the directory that contains the GPT model whose activations form the training data.
    - ckpt: The filename of the GPT model whose activations form the training data.
    - layer_name: The layer of the model whose outputs we want to disentangle. Options are:
        - `wte` [embedding layer]
        - `wpe` [positional encoding layer]
        - `attn{n}` [n-th layer's attention; n = 0, 1]
        - `mlp{n}` [n-th layer's mlp; n = 0, 1]
        - `res{n}` [n-th layer]
        - `ln_f` [final layer-norm before the LM-head]
    - batch_size: The batch size of the dataloader whose inputs generate the activations for the SAE.
    - exp_factor: The expansion factor of the SAE (the ratio of the hidden size to the input size).
    - alpha: If we want to use L1-regularization, the coefficient of the L1 norm of the latent in the loss function.
    - k: If we want to use top-k-regularization, the number of latent dimensions that are kept nonzero.
    - lr: The learning rate of the SAE. Used in an Adam optimizer.
    - train_iters: The number of iterations to train on.
    - val_iters: The number of iterations to validate on.
    - val_interval. The number of iterations after which we validate.
    - patience: If loss increases (not stays the same) for `patience` consecutive epochs, training is stopped.
- All of the above are optional arguments except `model_dir`, `ckpt`, and `layer_name`; furthermore, one of `alpha` and `k` must be provided.
- The dataset and GPT model are loaded according to the saved config file in the `model_dir`.
- Trained SAEs are saved in a subdirectory `sae_{i}` (depending on how many SAEs have been saved previously for the same model), which contains a `config.json` with the above arguments and a `model.pth` file.

# Model Checkpoints
- `results/scratch/12owob2t`: Model trained on prefix Expr. The SAEs are:
    - 0-215: $\alpha$-regularized sweep for `wte`. Best one 206.
    - 216-395: $k$-regularized sweep for `wte`. Best one 329.
    - 396-443: $\alpha$-regularized sweep for `res0`. Best one 427.
    - 444-447: The best two models of the two `wte` sweeps (206, 111, 113, 146 respectively), trained at max 5k iters with val_patience 5. Useless.
    - 448-487: $k$-regularized sweep for `res0`. Best one 454.
    - 488-489: The best model of the two `res0` sweeps (427, 454 respectively), trained at max 5k iters with val_patience 5.
    - 490-491: The best model of the two `wte` sweeps (206, 113 respectively), trained at max 5k iters with patience 5. Useless.
    - 492-493: The best model of the two `res0` sweeps (427, 454 respectively), trained at max 5k iters with patience 5. Useless.
    - 494-517: Sparsemax (old) sweep for `wte` and `res0` (over lr and exp_factor). Useless.
    - 518-529: Sparsemax (new, kds false) sweep for `res0` (over lr and exp_factor). Best 519 (65%).
    - 530-541: Sparsemax (new, kds true) sweep for `res0` (over lr and exp_factor). Best 541 (35%).
    - 542-553: Sparsemax (new, kds true, mode dist) sweep for `res0` (over lr and exp_factor).
- `results/scratch/3v4gwdfk`: Model trained on English (no prepositions, no relative clauses, p_conj 0.25 for nouns, 0.15 for verbs). The SAEs are:
    - 0-19: $k$-regularized sweep for `res0`. Useless.
    - 20-43: $\alpha$-regularized sweep for `res0`. Useless.
    - 44-63: $k$-regularized sweep for `res0`, trained with val_patience 3. Best 63, 61.
    - 64-87: $\alpha$-regularized sweep for `res0`, trained with val_patience 3. Best 67.
- `results/scratch/cpyib3ss`: Model trained on Dyck. The SAEs are:
    - 0-23: $\alpha$-regularized sweep for `res0`, with val_patience 3.
    - 24-43: $k$-regularized sweep for `res0`, with val_patience 3.
    - 44-63: $k$-regularized sweep for `res0`, with val_patience 3 and max 5k iters. Best 63.
    - 64-87: $\alpha$-regularized sweep for `res0`, with val_patience 3 and max 5k iters.
    - 88-107: $k$-regularized sweep for `res0`, using only depth-5 data.
- `results/scratch/vx8j11gp`: Model trained on English with transitivity and no other variations.
    - 0-47: $\alpha$-regularized sweep (twice). Best 47.
    - 48-67: $k$-regularized sweep. Best 62.
- `results/scratch/9rts35mx`: Model trained on English with adverbial and adjectival relative clauses.
    - 0-23: $\alpha$-regularized sweep. Useless.
    - 24-43: $k$-regularized sweep. Best 43.
- `results/scratch/bcb19qnd`: Model trained on English with adverbial and adjectival prepositional phrases (3 prepositions).
    - 0-23: $\alpha$-regularized sweep. Terrible.
    - 24-43: $k$-regularized sweep.
- `results/scratch/ktm5d2gn`: Model trained on English with the probability of conjunctions in NPs and VPs set to 0.3.
- `results/scratch/87bnn1o6`: Model trained on Dyck with the probability of nesting set to 30%. Average depth about 5.
    - 0-19: $k$-regularized sweep.
