"""Live terminal demo for the presentation: interpretability on a trained model.

Loads an already-trained LITE-MV (or LITE-MV-SE) checkpoint -- no training
happens here, only inference -- picks one correctly-classified test sample,
and narrates through the CAM / joint-saliency / fidelity story with printed
output suitable for reading aloud during a live demo.

Usage
-----
python3 scripts/demo_presentation.py \
    --root_path datasets/classification \
    --dataset_name KIMORE_clf_bn_Sq \
    --fold_number 1 \
    --estimator LITEMV \
    --run 0
"""

import argparse
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

from deep_rehab_pile.classifiers import (  # noqa: E402
    LITE_MV_CLASSIFIER,
    LITE_MV_SE_CLASSIFIER,
)
from deep_rehab_pile.utils import deletion_curve, load_classification_data  # noqa: E402

N_RANDOM_SEEDS_DEMO = 3


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="datasets/classification")
    parser.add_argument("--dataset_name", type=str, default="KIMORE_clf_bn_Sq")
    parser.add_argument("--fold_number", type=int, default=1)
    parser.add_argument(
        "--estimator", type=str, default="LITEMV", choices=["LITEMV", "LITEMV_SE"]
    )
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/")

    return parser.parse_args()


def section(title):
    """Print a narratable section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    """Run the live demo."""
    args = parse_args()

    section(f"Loading test data: {args.dataset_name}, fold {args.fold_number}")
    xtrain, ytrain, xtest, ytest, dataset_info = load_classification_data(
        dataset_name=args.dataset_name,
        root_path=args.root_path,
        fold_number=args.fold_number,
    )
    length_TS = dataset_info["length_TS"]
    n_joints = dataset_info["n_joints"]
    n_dim = dataset_info.get("dim", dataset_info.get("n_dim"))
    print(f"{len(xtest)} test sequences, {n_joints} joints x {n_dim} dims, "
          f"{length_TS} timesteps each.")

    output_dir_run = (
        f"{args.output_dir}classification/{args.dataset_name}/"
        f"fold{args.fold_number}/{args.estimator}/run{args.run}/"
    )
    cls = LITE_MV_CLASSIFIER if args.estimator == "LITEMV" else LITE_MV_SE_CLASSIFIER

    section(f"Loading trained checkpoint: {args.estimator}")
    t0 = time.time()
    classifier = cls(
        output_dir=output_dir_run,
        best_file_name="best_model",
        init_file_name="init_model",
        length_TS=length_TS,
        n_joints=n_joints,
        n_dim=n_dim,
    )
    classifier.n_classes = len(np.unique(np.concatenate((ytrain, ytest), axis=0)))

    preds, _ = classifier.predict(xtest)
    accuracy = (preds == ytest).mean()
    print(f"Loaded in {time.time() - t0:.1f}s. Test accuracy this run: {accuracy:.2f}")

    correct_idx = np.where(preds == ytest)[0]
    sample_idx = int(correct_idx[0]) if len(correct_idx) > 0 else 0
    xsample = xtest[sample_idx : sample_idx + 1]

    section(f"Explaining test sample #{sample_idx}")
    print(f"True label: {ytest[sample_idx]}   Predicted label: {preds[sample_idx]}")

    section("Per-joint saliency (which joint mattered)")
    saliency = classifier.compute_joint_saliency(xsample)[0]
    ranking = np.argsort(saliency)[::-1]
    print("Top-5 most important joints (index : normalized importance):")
    for j in ranking[:5]:
        print(f"  joint {j:2d} : {saliency[j]:.3f}")

    section("Temporal Grad-CAM (when in the exercise it mattered)")
    cam = classifier.compute_temporal_cam(xsample)[0]
    peak_t = int(np.argmax(cam))
    print(f"Peak importance at timestep {peak_t} / {length_TS} "
          f"({100 * peak_t / length_TS:.0f}% into the sequence).")

    section("Fidelity (deletion) test on this fold's test set")
    t0 = time.time()
    mean_saliency = classifier.compute_joint_saliency(xtest).mean(axis=0)
    joint_ranking = np.argsort(mean_saliency)[::-1]

    saliency_curve = deletion_curve(classifier, xtest, ytest, joint_ranking, n_dim=n_dim)

    rng = np.random.default_rng(seed=42)
    random_curves = np.zeros((N_RANDOM_SEEDS_DEMO, n_joints + 1))
    for s in range(N_RANDOM_SEEDS_DEMO):
        random_curves[s] = deletion_curve(
            classifier, xtest, ytest, rng.permutation(n_joints), n_dim=n_dim
        )
    random_curve = random_curves.mean(axis=0)

    print(f"Computed in {time.time() - t0:.1f}s.")
    print(f"{'k joints masked':>16} | {'saliency-ranked acc.':>21} | "
          f"{'random-ranked acc.':>19}")
    for k in [0, 1, 3, 5, n_joints]:
        print(f"{k:>16} | {saliency_curve[k]:>21.2f} | {random_curve[k]:>19.2f}")

    section("Done")
    print("Full figures (heatmaps, deletion curves) are in the report/slides.")


if __name__ == "__main__":
    main()
