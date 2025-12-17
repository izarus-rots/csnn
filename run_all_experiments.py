#!/usr/bin/env python3

import subprocess
import os
import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

EXPERIMENTS = [
    # Hyperparameter search — B-ALL
    ("tune_ball_cellcnn", "python train_cnn_cv.py config/tuning_ball_cellcnn.yaml"),
    ("tune_ball_cnn", "python train_cnn_cv.py config/tuning_ball_cnn.yaml"),
    ("tune_ball_logistic", "python train_logistic_cv.py config/tuning_ball_logistic.yaml"),
    ("tune_ball_reg", "python train_reg_cv.py config/tuning_ball_reg.yaml"),

    # Hyperparameter search — CLL
    ("tune_cll_cellcnn", "python train_cnn.py config/tuning_cll_cellcnn.yaml"),
    ("tune_cll_cnn", "python train_cnn.py config/tuning_cll_cnn.yaml"),
    ("tune_cll_logistic", "python train_logistic.py config/tuning_cll_logistic.yaml"),
    ("tune_cll_reg", "python train_reg.py config/tuning_cll_reg.yaml"),

    # Test set evaluation — B-ALL
    ("best_ball_cellcnn", "python train_cnn.py config/best_ball_cellcnn.yaml"),
    ("best_ball_cnn", "python train_cnn.py config/best_ball_cnn.yaml"),
    ("best_ball_logistic", "python train_logistic.py config/best_ball_logistic.yaml"),
    ("best_ball_reg", "python train_reg.py config/best_ball_reg.yaml"),

    # Test set evaluation — CLL
    ("best_cll_cellcnn", "python train_cnn.py config/best_cll_cellcnn.yaml"),
    ("best_cll_cnn", "python train_cnn.py config/best_cll_cnn.yaml"),
    ("best_cll_logistic", "python train_logistic.py config/best_cll_logistic.yaml"),
    ("best_cll_reg", "python train_reg.py config/best_cll_reg.yaml"),

    # No-init ablations
    ("ablate_noinit_ball_logistic", "python train_logistic_ablation_no_init.py config/best_ball_logistic.yaml"),
    ("ablate_noinit_ball_reg", "python train_reg_ablation_no_init.py config/best_ball_reg.yaml"),
    ("ablate_noinit_cll_logistic", "python train_logistic_ablation_no_init.py config/best_cll_logistic.yaml"),
    ("ablate_noinit_cll_reg", "python train_reg_ablation_no_init.py config/best_cll_reg.yaml"),

    # Init-only ablations
    ("ablate_init_only_ball", "python train_ablation_init_only.py config/best_ball_reg.yaml"),
    ("ablate_init_only_cll", "python train_ablation_init_only.py config/best_cll_reg.yaml"),
]

SUMMARY_PATH = os.path.join(LOG_DIR, "summary.txt")

with open(SUMMARY_PATH, "w") as summary:
    summary.write(f"Experiment run started: {datetime.datetime.now()}\n\n")

    for name, cmd in EXPERIMENTS:
        out_path = os.path.join(LOG_DIR, f"{name}.out")
        err_path = os.path.join(LOG_DIR, f"{name}.err")

        summary.write(f"Running: {name}\n")
        summary.flush()

        with open(out_path, "w") as fout, open(err_path, "w") as ferr:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=fout,
                stderr=ferr,
            )
            returncode = process.wait()

        status = "OK" if returncode == 0 else f"FAIL (code {returncode})"
        summary.write(f"  Status: {status}\n\n")
        summary.flush()

print("All experiments completed. See logs/ for details.")
