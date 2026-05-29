# Hyperspectral Image Classification using Modern Architectures and Few-Shot Learning Frameworks

A comprehensive, state-of-the-art framework for high-performance classification of hyperspectral images (HSI). This repository implements modern neural network architectures—including **1D/2D Convolutional Neural Networks (CNNs)**, **Multi-Head Self-Attention**, and **State Space Models (Mamba)**—integrated within advanced **Few-Shot Learning (FSL)** paradigms.

---

## 🚀 Overview

Hyperspectral images capture detailed spectral information across hundreds of contiguous bands, but labeled datasets are often small and expensive to acquire. This codebase addresses the challenge using **Few-Shot Learning**, enabling models to generalize to new classes with very few labeled samples (e.g., 5-shot learning).

### 🧠 Modern Architectures
1. **CNN (Baseline)**: Efficient local feature extraction across spectral bands.
2. **CNN + Self-Attention**: Integrates Multi-Head Attention to capture global dependencies and band correlations.
3. **CNN + Mamba (State Space Model)**: Employs a customized **Stable Mamba Block** for linear-complexity sequence modeling of spectral profiles.
4. **Hybrid CNN + Attention + Mamba**: SOTA fusion capturing local patterns, global context, and sequential band transitions.

### 🔬 Few-Shot Learning Frameworks
* **Prototypical Networks (ProtoNet)**: Computes prototypical representations for each class in embedding space and classifies query samples using Euclidean distance.
* **Model-Agnostic Meta-Learning (MAML)**: Optimization-based meta-learning that initializes weights for fast adaptation in a few gradient steps.
* **Relation Networks (RelationNet)**: Learns a deep non-linear metric to calculate relation (similarity) scores between support and query embeddings.
* **Siamese Networks**: Implements pairwise similarity learning using contrastive objectives.

---

## 📂 Repository Structure

```directory
├── fsl_protonet.ipynb           # ProtoNet training & analysis notebook
├── fsl_maml.ipynb               # MAML training & analysis notebook
├── fsl_relationnet.ipynb        # RelationNet training & analysis notebook
├── fsl_siemese.ipynb            # Siamese network pipeline notebook
├── hyper.ipynb                  # Initial HSI preprocessing & baseline notebook
├── train_protonet_all.py        # Automated ProtoNet trainer (all architectures)
├── train_maml_all.py            # Automated MAML trainer (all architectures)
├── train_relationnet_all.py     # Automated RelationNet trainer (all architectures)
├── train_siamese_all.py         # Automated Siamese trainer (all architectures)
├── tune_protonet_all.py         # Hyperparameter tuning script for ProtoNet
├── evaluate_best_weights.py     # Model evaluation and comparison runner
├── cnn_mamba_protonet_diagram.svg # Architecture and training workflow visualization
├── .gitignore                   # Standard Python, data, & weights exclusions
└── README.md                    # Project documentation
```

---

## 🛠️ Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure you have TensorFlow 2.x and other ML libraries installed:
   ```bash
   pip install tensorflow numpy scikit-learn matplotlib pandas ipykernel
   ```

---

## 📊 Training the Models

You can run individual Jupyter Notebooks for interactive step-by-step training, or use the automated training scripts to train all architectures (`cnn`, `cnn_attention`, `cnn_mamba`, `cnn_attention_mamba`) under specific FSL frameworks:

### Prototypical Networks
```bash
python train_protonet_all.py
```

### Model-Agnostic Meta-Learning (MAML)
```bash
python train_maml_all.py
```

### Relation Networks
```bash
python train_relationnet_all.py
```

### Siamese Networks
```bash
python train_siamese_all.py
```

---

## 📈 Evaluation and Results

Run the evaluation suite to compare test metrics (Mean, Max, Min Accuracies) across different architecture and framework combinations:

```bash
python evaluate_best_weights.py
```

---

## 📝 License
This project is licensed under the MIT License.
