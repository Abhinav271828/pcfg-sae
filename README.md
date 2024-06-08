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