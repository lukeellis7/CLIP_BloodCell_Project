**data_loader.py**

Loads the dataset, resolves missing files, filters out ultra-rare classes, and splits everything into train/val/test with stratified sampling.
Also prints class distribution so you can sanity-check balance before training.

**train.py**

Trains a linear probe (a single classifier head) on top of a frozen CLIP image encoder.
Includes:

augmentations (light flips/rotations)

class-balanced sampling

early stopping

AMP for mixed-precision
At the end, it saves the best classifier + CLIP processor + labels for later inference.

**clip_inference.py**

Loads the trained model (or falls back to zero-shot CLIP) and runs predictions on the whole dataset.
Outputs:

per-image CSV with predicted label(s), confidences, and Top-1/Top-2 correctness flags

printed accuracy summary

**evaluation.py**

Crunches metrics and makes visuals:

prints Top-1 / Top-2 accuracy

saves a confusion matrix (counts or normalized %)

writes a detailed classification report

runs a confidence threshold sweep to show accuracy vs coverage trade-off
Everything lands in the ../results folder ready to drop into the report.

**utils.py**

A couple of plotting/report helpers that evaluation.py imports:
