
# WMMSE Deep Unfolding (refactored from notebook)

This folder is a small refactor of the uploaded `Deep_Unfolded_WMMSE_versus_WMMSE.ipynb` into runnable Python modules.

## What you get
- `wmmse_unfolded_project/utils.py`:
  Numpy baselines (WMMSE/ZF/RZF) + TF ops used by the unfolded graph.
- `wmmse_unfolded_project/models/unfolded_graph.py`:
  Builds the TF1-style computation graph for the unfolded method.
- `wmmse_unfolded_project/scripts/train_unfolded.py`:
  Trains the unfolded model and saves a checkpoint.
- `wmmse_unfolded_project/scripts/infer_unfolded.py`:
  Loads a saved checkpoint and runs inference.

## How to run
From the directory that contains `wmmse_unfolded_project/`:

### Train (saves weights)
```bash
python -m wmmse_unfolded_project.scripts.train_unfolded
```

Optionally set an output directory:
```bash
OUT_DIR=./checkpoints python -m wmmse_unfolded_project.scripts.train_unfolded
```

### Inference (loads weights)
```bash
CKPT_PATH=./checkpoints/unfolded_wmmse.ckpt python -m wmmse_unfolded_project.scripts.infer_unfolded
```

## TensorFlow note
This code uses TF1 graph execution via `tf.compat.v1` to match the notebook.
