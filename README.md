Breast Cancer Classification – PyTorch & TensorFlow Implementations
📌 Overview
This project demonstrates end-to-end implementation of a Breast Cancer classification model using two different deep learning frameworks – PyTorch and TensorFlow/Keras.
It’s designed as a learning exercise to:

Understand the data preprocessing pipeline for ML/DL projects.

Learn model building, training, and evaluation in both frameworks.

Compare workflow similarities and differences between PyTorch and TensorFlow.

Strengthen skills in binary classification problems.

The dataset used is the Breast Cancer Wisconsin dataset from scikit-learn.

🧠 Learning Goals
Through this project, you will learn:

Data Handling

Loading datasets from sklearn.datasets.

Exploring dataset features & targets.

Creating pandas DataFrames for better data visualization.

Splitting into training and testing sets.

Data Preprocessing

Standardizing features using StandardScaler for PyTorch implementation.

Ensuring feature scaling for better convergence.

Model Development

PyTorch: Building custom neural networks with torch.nn.Module.

TensorFlow/Keras: Using Sequential models and Dense layers.

Activation functions (ReLU, Sigmoid) for classification tasks.

Model Training

Defining loss functions (BCELoss in PyTorch, binary_crossentropy in Keras).

Choosing optimizers (Adam) and tuning learning rates.

Understanding the training loop (manual in PyTorch vs built-in .fit() in Keras).

Model Evaluation

Accuracy calculation on both training and test datasets.

Interpreting binary classification results.

Comparison between Frameworks

PyTorch offers more control over the training loop.

TensorFlow/Keras offers ease of use with high-level APIs.

🗂️ Project Structure
bash
Copy
Edit
📦 Breast-Cancer-Classification
 ┣ 📜 test_torch.ipynb     # PyTorch implementation
 ┣ 📜 test_tf.ipynb        # TensorFlow/Keras implementation
 ┣ 📜 README.md            # This file
🛠️ Technologies Used
Common Libraries:
Python 3.x

NumPy – Numerical operations

Pandas – Data handling

scikit-learn – Dataset loading, preprocessing, and splitting

Matplotlib – Data visualization

PyTorch Version (test_torch.ipynb)
PyTorch (torch, torch.nn, torch.optim) – Model creation, training, and evaluation

TensorFlow Version (test_tf.ipynb)
TensorFlow/Keras – High-level neural network implementation

📊 Dataset Details
Name: Breast Cancer Wisconsin Dataset
Source: sklearn.datasets.load_breast_cancer()

Features: 30 numeric features (mean radius, mean texture, etc.)

Target: Binary classification –

0 → Malignant

1 → Benign

Samples: 569

🚀 How to Run
1️⃣ Install Requirements
bash
Copy
Edit
pip install torch tensorflow scikit-learn pandas matplotlib
2️⃣ Run PyTorch Version
bash
Copy
Edit
jupyter notebook test_torch.ipynb
3️⃣ Run TensorFlow Version
bash
Copy
Edit
jupyter notebook test_tf.ipynb
📚 What I Learned
How to prepare and preprocess tabular datasets for deep learning models.

How binary classification works with neural networks.

The importance of normalization/standardization before training.

The difference between manual training loops (PyTorch) and automated training APIs (Keras).

How to interpret classification accuracy and avoid overfitting.

🔮 Future Improvements
Add cross-validation to ensure model generalization.

Implement regularization techniques (Dropout, L2 regularization).

Visualize loss and accuracy curves for deeper analysis.

Extend to multi-class classification tasks.

🏷️ License
This project is for educational purposes only. Feel free to fork, modify, and use it to enhance your learning.
