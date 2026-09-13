"""Generate CAM/saliency figures and a deletion-test fidelity curve.

Loads an already-trained LITEMV (or LITEMV_SE) model for one dataset/fold
(produced by main.py) and:
  - saves a temporal Grad-CAM heatmap and a per-joint saliency bar plot for
    a handful of test samples,
  - runs the deletion test (saliency-ranked vs. random-ranked joint
    masking) and saves the resulting accuracy curves as a CSV and a plot.

Usage
-----
python3 scripts/run_interpretability.py \
    --root_path datasets/classification \
    --dataset_name KIMORE_clf_bn_Sq \
    --fold_number 0 \
    --estimator LITEMV \
    --run 0 \
    --output_dir results/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from deep_rehab_pile.classifiers import LITE_MV_CLASSIFIER, LITE_MV_SE_CLASSIFIER
from deep_rehab_pile.utils import deletion_curve, load_classification_data

N_RANDOM_SEEDS = 10


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="datasets/classification")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--fold_number", type=int, default=0)
    parser.add_argument(
        "--estimator", type=str, default="LITEMV", choices=["LITEMV", "LITEMV_SE"]
    )
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--n_examples", type=int, default=3)

    return parser.parse_args()


def _load_classifier(args, output_dir_run, length_TS, n_joints, n_dim):
    common_kwargs = {
        "output_dir": output_dir_run + "/",
        "best_file_name": "best_model",
        "init_file_name": "init_model",
        "length_TS": length_TS,
        "n_joints": n_joints,
        "n_dim": n_dim,
    }

    if args.estimator == "LITEMV":
        return LITE_MV_CLASSIFIER(**common_kwargs)

    return LITE_MV_SE_CLASSIFIER(**common_kwargs)


def main():
    """Run the interpretability analysis."""
    args = parse_args()

    xtrain, ytrain, xtest, ytest, dataset_info = load_classification_data(
        dataset_name=args.dataset_name,
        root_path=args.root_path,
        fold_number=args.fold_number,
    )

    length_TS = dataset_info["length_TS"]
    n_joints = dataset_info["n_joints"]
    try:
        n_dim = dataset_info["dim"]
    except KeyError:
        n_dim = dataset_info["n_dim"]

    output_dir_run = os.path.join(
        args.output_dir,
        "classification",
        args.dataset_name,
        "fold" + str(args.fold_number),
        args.estimator,
        "run" + str(args.run),
    )

    classifier = _load_classifier(args, output_dir_run, length_TS, n_joints, n_dim)
    classifier.n_classes = len(np.unique(np.concatenate((ytrain, ytest), axis=0)))

    cams = classifier.compute_temporal_cam(xtest)
    saliency = classifier.compute_joint_saliency(xtest)

    out_dir = output_dir_run
    n_examples = min(args.n_examples, len(xtest))

    fig, axes = plt.subplots(n_examples, 1, figsize=(8, 3 * n_examples), squeeze=False)
    for i in range(n_examples):
        axes[i][0].plot(cams[i])
        axes[i][0].set_title(f"Temporal CAM - test sample {i} (label={ytest[i]})")
        axes[i][0].set_xlabel("timestep")
        axes[i][0].set_ylabel("importance")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "temporal_cam_examples.pdf"))
    plt.close(fig)

    mean_saliency = saliency.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(n_joints), mean_saliency)
    ax.set_title(f"Mean joint saliency - {args.dataset_name} - {args.estimator}")
    ax.set_xlabel("joint index")
    ax.set_ylabel("mean importance")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "joint_saliency.pdf"))
    plt.close(fig)

    joint_ranking_saliency = np.argsort(mean_saliency)[::-1]

    saliency_curve = deletion_curve(
        classifier, xtest, ytest, joint_ranking_saliency, n_dim=n_dim
    )

    rng = np.random.default_rng(seed=42)
    random_curves = np.zeros(shape=(N_RANDOM_SEEDS, n_joints + 1))
    for seed in range(N_RANDOM_SEEDS):
        random_ranking = rng.permutation(n_joints)
        random_curves[seed] = deletion_curve(
            classifier, xtest, ytest, random_ranking, n_dim=n_dim
        )
    random_curve_mean = random_curves.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(saliency_curve, label="saliency-ranked deletion", marker="o")
    ax.plot(random_curve_mean, label="random-ranked deletion (mean)", marker="x")
    ax.set_title(f"Deletion test - {args.dataset_name} - {args.estimator}")
    ax.set_xlabel("number of top joints masked")
    ax.set_ylabel("accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "deletion_curve.pdf"))
    plt.close(fig)

    header = "k,saliency_accuracy,random_accuracy_mean"
    rows = np.column_stack((np.arange(n_joints + 1), saliency_curve, random_curve_mean))
    np.savetxt(
        os.path.join(out_dir, "deletion_curve.csv"),
        rows,
        delimiter=",",
        header=header,
        comments="",
    )


if __name__ == "__main__":
    main()
