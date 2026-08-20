"""
Optuna hyperparameter tuning for PIPNet on GMDB.

Strategy
--------
Stage 1 (prototype pretraining) is expensive and relatively robust — run it
once and save net_pretrained. This script only tunes Stage 2 hyperparameters,
loading that pretrained checkpoint for every trial.

Usage
-----
    python tune.py                        # run 20 trials sequentially
    python tune.py --n_trials 40          # more trials
    python tune.py --study_name my_study  # resume an existing study

The Optuna study is stored in an SQLite database so trials survive job
timeouts and can be resumed by resubmitting tune_job.sh.
"""

import argparse
import os
import subprocess
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths — adjust if needed
# ---------------------------------------------------------------------------
PRETRAINED_CKPT = (
    '/data2/masino_lab/abamini/pipnet/runs/run_6_GMDN_sysndromes_with_bias_weighted_loss/checkpoints/net_pretrained'
)
LOG_BASE   = '/data2/masino_lab/abamini/pipnet/runs/optuna'
PIPNET_DIR = '/home/abamini/protogestalt'

# Metric columns per task type — detected automatically from the CSV header
# Binary (2-class):  test_f1   (F1 is more meaningful for imbalanced data)
# Multi-class:       test_top1_acc
BINARY_METRIC     = 'test_f1'
MULTICLASS_METRIC = 'test_top1_acc'


def detect_metric_col(csv_path: str) -> str:
    """Read the CSV header and return the right metric column for this run."""
    try:
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
        return BINARY_METRIC if BINARY_METRIC in header else MULTICLASS_METRIC
    except Exception:
        return MULTICLASS_METRIC


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial, dataset: str = 'gmdb') -> float:
    # --- suggest hyperparameters -------------------------------------------
    lr        = trial.suggest_float('lr',           0.01,  0.1,   log=True)
    lr_block  = trial.suggest_float('lr_block',     1e-4,  1e-3,  log=True)
    lr_net    = trial.suggest_float('lr_net',       1e-4,  1e-3,  log=True)
    freeze_ep = trial.suggest_int(  'freeze_epochs', 5,    20)
    epochs    = trial.suggest_int(  'epochs',        20,   100)
    bias      = trial.suggest_categorical('bias',    [True, False])
    w_loss    = trial.suggest_categorical('weighted_loss', [True, False])
    # Region-masked pooling is enabled for every trial; tune the region radius
    # (fraction of the normalized image). Small = tight regions (closer to a hard
    # landmark constraint), large = permissive regions (closer to global pooling).
    mask_radius = trial.suggest_float('mask_radius', 0.08, 0.30)

    # --- build paths -------------------------------------------------------
    run_num  = f'trial_{trial.number}'
    # args.py appends _{run_number} to log_dir automatically
    log_dir_base   = os.path.join(LOG_BASE, 'run_pipnet')
    actual_log_dir = f'{log_dir_base}_{run_num}'
    csv_path       = os.path.join(actual_log_dir, 'log_epoch_overview.csv')

    # --- build command -----------------------------------------------------
    cmd = [
        'python', 'main.py',
        '--dataset',            dataset,
        '--state_dict_dir_net', PRETRAINED_CKPT,
        '--epochs_pretrain',    '0',
        '--epochs',             str(epochs),
        '--lr',                 str(lr),
        '--lr_block',           str(lr_block),
        '--lr_net',             str(lr_net),
        '--freeze_epochs',      str(freeze_ep),
        '--log_dir',            log_dir_base,
        '--run_number',         run_num,
        '--num_workers',        '8',
        '--masked_pool',                  # region-masked pooling (tuned radius below)
        '--mask_radius',        str(mask_radius),
        '--no_vis',                       # skip all image saving — tuning only needs accuracy
    ]
    if bias:
        cmd.append('--bias')
    if w_loss:
        cmd.append('--weighted_loss')

    print(f'\n[Trial {trial.number}] Params: lr={lr:.5f}, lr_block={lr_block:.5f}, '
          f'lr_net={lr_net:.5f}, freeze_epochs={freeze_ep}, '
          f'epochs={epochs}, bias={bias}, weighted_loss={w_loss}, '
          f'mask_radius={mask_radius:.3f}', flush=True)

    # --- run training ------------------------------------------------------
    try:
        subprocess.run(cmd, check=True, cwd=PIPNET_DIR)
    except subprocess.CalledProcessError as e:
        print(f'[Trial {trial.number}] Training failed: {e}', flush=True)
        return 0.0

    # --- read metric from CSV ---------------------------------------------
    try:
        df         = pd.read_csv(csv_path)
        metric_col = detect_metric_col(csv_path)
        # Keep only numeric epoch rows (skip pretrain n.a. rows and purity rows)
        df = df[pd.to_numeric(df['epoch'], errors='coerce').notna()]
        df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
        best = df[metric_col].dropna().max()
        print(f'[Trial {trial.number}] Best {metric_col}: {best:.4f}', flush=True)
        return float(best)
    except Exception as e:
        print(f'[Trial {trial.number}] Could not read CSV ({e})', flush=True)
        return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials',   type=int, default=20)
    parser.add_argument('--study_name', type=str, default='pipnet_gmdb')
    parser.add_argument('--dataset',    type=str, default='gmdb',
                        help='Dataset name passed to main.py (e.g. gmdb). '
                             'Metric is auto-detected: test_f1 for 2-class, '
                             'test_top1_acc for multi-class.')
    args = parser.parse_args()

    os.makedirs(LOG_BASE, exist_ok=True)
    db_path = os.path.join(LOG_BASE, f'{args.study_name}.db')

    study = optuna.create_study(
        direction='maximize',
        storage=f'sqlite:///{db_path}',
        study_name=args.study_name,
        load_if_exists=True,          # resume if job was interrupted
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    study.optimize(lambda trial: objective(trial, dataset=args.dataset),
                   n_trials=args.n_trials)

    print('\n===== Best trial =====')
    print('Value :', study.best_value)
    print('Params:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')

    # Save summary
    summary_path = os.path.join(LOG_BASE, f'{args.study_name}_results.csv')
    study.trials_dataframe().to_csv(summary_path, index=False)
    print(f'\nAll trials saved to {summary_path}')


if __name__ == '__main__':
    main()
