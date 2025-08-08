# 🧬 Breast Cancer Classification – PyTorch & TensorFlow

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)
![TensorFlow](https://img.shields.io/badge/TensorFlow-orange?logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-lightgrey?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview
This project is a **side-by-side implementation** of a **Breast Cancer Classification model** using two leading deep learning frameworks — **PyTorch** and **TensorFlow/Keras**.  
The idea was to **learn by doing** and compare both frameworks in terms of:
- Syntax & workflow
- Training loop control vs. abstraction
- Performance and ease of implementation

The dataset used is the **Breast Cancer Wisconsin Dataset**, a classic binary classification dataset available in `scikit-learn`.

---

## 🎯 Objectives
- Learn how to **load, preprocess, and split** datasets for deep learning.
- Understand how **binary classification** works in neural networks.
- Compare **manual training loops** in PyTorch vs **high-level APIs** in TensorFlow/Keras.
- Strengthen knowledge of **loss functions**, **optimizers**, and **evaluation metrics**.
- Build a project that can be **showcased in a portfolio**.

---

## 🛠️ Technologies & Libraries

### Common:
- **Python 3.x**
- `numpy`, `pandas` – Data manipulation
- `scikit-learn` – Dataset loading, train-test splitting, standardization
- `matplotlib` – Visualization

### PyTorch Implementation (`test_torch.ipynb`):
- `torch` – Core deep learning library
- `torch.nn` – Neural network layers
- `torch.optim` – Optimizers (`Adam`)
- Manual training loop for full control

### TensorFlow Implementation (`test_tf.ipynb`):
- `tensorflow` / `keras` – Model building and training
- `Sequential` API for rapid prototyping
- Built-in `.fit()` training loop

---

## 📊 Dataset Information
**Dataset Name:** Breast Cancer Wisconsin Dataset  
**Samples:** 569  
**Features:** 30 numerical features (mean radius, texture, perimeter, etc.)  
**Target:**  
- `0` → Malignant  
- `1` → Benign  

---

## 🧠 Model Architecture

**Layers:**
- **Input Layer:** 30 neurons (features)
- **Hidden Layer(s):** Dense with `ReLU` activation
- **Output Layer:** Single neuron with `Sigmoid` activation

---

##🧠 Learning Journey
- **Data Preprocessing:** Scaling data improves training convergence.

- **Model Design:** Input → Hidden (ReLU) → Output (Sigmoid).

- **Training:** Manual loops (PyTorch) vs .fit() abstraction (Keras).

- **Loss Functions:** BCELoss (PyTorch) vs binary_crossentropy (Keras).

- **Evaluation:** Accuracy metrics to validate performance.

---

##📊 Results & Accuracy Comparison
| **Framework**            | **Training Accuracy** | **Testing Accuracy** |
| -------------------- | ----------------- | ---------------- |
| **PyTorch**          | 98.2%             | 97.1%            |
| **TensorFlow/Keras** | 98.6%             | 97.5%            |

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/e6aa93ba-cd57-4ee4-9c2a-d32c369bdea8" />


✅ Both frameworks performed very similarly, proving that model architecture and preprocessing choices often matter more than the framework itself.
**PyTorch** gave more training control, while **TensorFlow/Keras** allowed faster prototyping.

---
##🔍 PyTorch vs TensorFlow – Key Takeaways

| Feature         | PyTorch                   | TensorFlow/Keras                          |
| --------------- | ------------------------- | ----------------------------------------- |
| **Control**     | ✅ High (manual loops)     | ⚠️ Lower (abstracted)                     |
| **Ease of Use** | ⚠️ Steeper learning curve | ✅ Beginner-friendly                       |
| **Flexibility** | ✅ Customizable            | ⚠️ Requires subclassing for customization |
| **Community**   | Large                     | Large                                     |

---

