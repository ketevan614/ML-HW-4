# FER2013 Facial Expression Recognition

Homework 4 for the Kaggle task **Challenges in Representation Learning: Facial Expression Recognition Challenge**.

Goal: classify 48x48 grayscale face images into seven emotions:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

The main homework file is the notebook:

```text
notebooks/fer2013_homework.ipynb
```

It has the work in one place: data check, models, forward/backward sanity checks, training, Wandb logging, and comparison.

Notebook-first because this is homework, not a production package.

## Dataset

The original competition download was annoying, so I used the FER2013 Kaggle mirror:

```powershell
kaggle datasets download -d msambare/fer2013 -p data
powershell -Command "Expand-Archive -Path data\fer2013.zip -DestinationPath data\fer2013"
```

Expected data layout:

```text
data/fer2013/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  test/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
```

The `data/` folder is ignored by Git and should not be committed.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
wandb login
```

## Sanity Checks

Before training, the notebook checks one batch:

- loads one batch
- checks the model output shape
- computes cross entropy loss
- runs `loss.backward()`
- verifies gradients exist and are finite

If this fails, training for many epochs is pointless.

Most notebook cells run fast because they only import packages, define functions, or count files. Real training starts only when `RUN_ALL_EXPERIMENTS = True`.

## Experiments

I use three models:

| Run | Architecture | Main idea | Why |
| --- | --- | --- | --- |
| `tiny_cnn_lr1e-3` | TinyCNN | Two conv layers | Baseline |
| `better_cnn_aug_dropout03` | BetterCNN | More conv blocks + BatchNorm + Dropout | Stronger CNN |
| `residual_cnn_adamw_lr3e-4` | ResidualCNN | Small residual model | Test skip connections |

Notebook-first workflow:

```powershell
jupyter notebook notebooks/fer2013_homework.ipynb
```

## Wandb Tracking

Each architecture and hyperparameter setting should be a separate Wandb run.

Logged values:

- train loss
- train accuracy
- validation/test loss
- validation/test accuracy
- gradients
- config/hyperparameters
- confusion matrix
- best checkpoint path

Project name:

```text
fer2013-homework
```

## Results

Fill this table after running experiments.

| Run | Learning rate | Dropout | Augmentation | Best validation accuracy | Notes |
| --- | ---: | ---: | --- | ---: | --- |
| `tiny_cnn_lr1e-3` | 0.001 | 0.25 | no | TODO | Baseline |
| `better_cnn_aug_dropout03` | 0.001 | 0.30 | yes | TODO | More capacity and regularization |
| `residual_cnn_adamw_lr3e-4` | 0.0003 | 0.40 | yes | TODO | Residual architecture |

## Notes

First model is small on purpose. Second model adds more layers/regularization. Third model tries residual blocks.

After training, I will add the actual scores and Wandb links here.
