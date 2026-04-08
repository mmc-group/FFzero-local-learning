# FFzero-local-learning

This repository contains the implementation code for paper: Guo et al., [Local learning for stable backpropagation-free neural network training towards physical learning](https://arxiv.org/abs/2603.24790), _Arxiv:2603.24790._

---

## Repository Structure

```
.
├── script/
│   ├── main.py          # MLP/CNN training on MNIST, FashionMNIST, synthetic functions with training paradigms including BP+AD, BP+DD, FF+AD and FF+DD(FFzero)
│   └── main_onn.py      # Optical Neural Network (ONN) training on MNIST with training paradigms including BP+AD, BP+DD, FF+AD and FF+DD(FFzero)
├── core/
│   ├── data_loader.py    # Dataset loading and preprocessing
│   ├── solver.py         # Training solvers (FF+AD, FF+DD, BP+AD, BP+DD)
│   ├── core_onn.py       # Optical neural network model and Optical mesh logic
│   └── onn_viz.py        # Optical model visualization utilities
└── results/              # Experiment logs and plots (auto-generated)
```

## Installation

To install dependencies, run

```
micromamba env create -f environment.yml
```


## Usage

### 1. MLP / CNN Training (`main.py`)

Train MLP or CNN models on MNIST, FashionMNIST, or synthetic functions using one of four solvers.

```bash
cd script
python main.py [OPTIONS]
```

**Key Arguments**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `MNIST` | Dataset: `MNIST`, `FashionMNIST`, `function1`, `function2` |
| `--task` | `classification` | Task: `classification` or `regression` |
| `--model` | `cnn` | Model type: `mlp` or `cnn` |
| `--solver` | `ff_dd` | Solver: `ff_ad`, `ff_dd`, `bp_ad`, `bp_dd` |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `256` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--eps` | `1e-3` | Epsilon for DD optimization |
| `--num_directions` | `1` | Directions sampled per DD step |

**MLP-specific Arguments**

| Argument | Default | Description |
|---|---|---|
| `--mlp_ref_dim` | `[100,100,100,10]` | Layer dimensions (same as prototype vectors) |

**CNN-specific Arguments**

| Argument | Default | Description |
|---|---|---|
| `--cnn_channels` | `[16, 16]` | Convolutional channels per cnn layer |
| `--cnn_kernel_size` | `[6, 6]` | Kernel sizes |
| `--cnn_ref_dim` | `[10, 10]` | Dimensions of prototype vectors in convolution layer |
| `--cnn_pooling_size` | `[2, 2]` | Pooling sizes |
| `--cnn_fc_dim` | `[10]` | Dimension of FC layer |
| `--cnn_fc_ref_dim` | `[10]` | Dimensions of prototype vectors in FC layer |

**Examples**

```bash
# CNN with FF+DD on MNIST
python main.py --model cnn --solver ff_dd --dataset MNIST --epochs 10

# Regression on a synthetic function
python main.py --task regression --dataset function1 --model mlp --solver ff_ad
```

---

### 2. Optical Neural Network(ONN) Training (`main_onn.py`)

Train a photonic neural network (based on Neurophox: https://github.com/solgaardlab/neurophox) on MNIST using TensorFlow and one of four solvers.

```bash
cd script
python main_onn.py [OPTIONS]
```

**Key Arguments**

| Argument | Default | Description |
|---|---|---|
| `--solver` | `ff_dd` | Solver: `ff_ad`, `ff_dd(FFzero)`, `bp_ad`, `bp_dd` |
| `--crop_size` | `28` | Input crop size (n×n pixels → n² ports) |
| `--n_layers` | `2` | Number of photonic mesh layers |
| `--epochs` | `100` | Training epochs |
| `--batch_size` | `128` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--eps` | `1e-3` | Epsilon for DD optimization |
| `--num_directions` | `1` | Directions sampled per DD step |

**Examples**

```bash
# Train a 2-layer ONN with FF+DD
python main_onn.py --solver ff_dd --n_layers 2 --epochs 50

# Train with BP+AD on a smaller crop
python main_onn.py --solver bp_ad --crop_size 14
```

---

## Solvers

| Solver | Description |
|---|---|
| `ff_ad` | Forward-Forward algorithm with automatic differentiation |
| `ff_dd` | Forward-Forward algorithm with directional derivative optimization (FFzero) |
| `bp_ad` | Backpropagation with automatic differentiation |
| `bp_dd` | Backpropagation with directional derivative optimization |

The **forward-forward(ff)** solvers train each layer independently using a contrastive positive/negative pass and cosine similarity to simplex prototype vectors. The **directional derivative(dd)** solvers estimate gradients by perturbing weights along random directions, avoiding backpropagation through the model.

---

## Outputs

Experiment results are saved automatically under `results/`:

- **MLP/CNN**: `results/log-{dataset}-{task}-{model}-{solver}-{timestamp}/`
  - `log.txt` — training logs
  - Accuracy and loss plots

- **ONN**: `results/onn_{solver}_{crop_size}_{n_layers}_layers/`
  - Phase parameter visualizations (before/after training)
  - Accuracy/loss curves

---

## Notes

- The FF solvers require an output dimension of 10 (for cosine similarity to simplex prototype vectors); BP solvers use output dim 1 for regression tasks. This is handled automatically.
- For ONN training, model parameters are the optical phase angles (θ, φ, γ) of each beamsplitter in the photonic mesh.
