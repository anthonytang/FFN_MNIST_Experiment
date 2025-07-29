# FFN MNIST Experiment

This repo compares FFN_GeGLU and FFN_ReLU on MNIST.

## Requirements
Install dependencies using `uv`:
```bash
uv venv
uv pip install -r requirements.txt
```

## To Run
```bash
python train.py
```

## Description
- Compares GeGLU vs ReLU FFN on MNIST
- Uses PyTorch Lightning
- Random hyperparameter search
- Plots accuracy vs hidden dim with error bars (bootstrap)
