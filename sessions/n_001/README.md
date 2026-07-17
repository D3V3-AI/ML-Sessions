# ML SessionS #1 — MNIST

## About

Multi-class classification problem: recognizing handwritten digits (0-9) from 28x28 grayscale images. This session builds a simple neural network (784 → 128 → 10) from scratch with NumPy — no deep learning framework — implementing forward pass, backpropagation, and gradient descent manually.

## Dataset

- **Source:** [MNIST database](http://yann.lecun.com/exdb/mnist/), loaded via `tensorflow.keras.datasets.mnist` (downloaded at runtime, not stored locally)
- 60,000 training images and 10,000 test images, each 28x28 pixels, labeled with the digit (0-9) they represent.

## Contents

```
n_001/
├── mnist.py             # standalone script: NN implemented from scratch (no ML libraries)
├── mnist_nb.ipynb        # notebook version of the from-scratch implementation
├── session_1.ipynb       # main session notebook — theory, math derivations, and training
└── _mnist.ipynb          # early/scratch notebook
```
