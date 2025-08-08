🧬 Breast Cancer Classification – PyTorch & TensorFlow





📖 Overview
This project is a side-by-side implementation of a Breast Cancer Classification model using two leading deep learning frameworks — PyTorch and TensorFlow/Keras.
The idea was to learn by doing and compare both frameworks in terms of:

Syntax & workflow

Training loop control vs. abstraction

Performance and ease of implementation

The dataset used is the Breast Cancer Wisconsin Dataset, a classic binary classification dataset available in scikit-learn.

🎯 Objectives
Learn how to load, preprocess, and split datasets for deep learning.

Understand how binary classification works in neural networks.

Compare manual training loops in PyTorch vs high-level APIs in TensorFlow/Keras.

Strengthen knowledge of loss functions, optimizers, and evaluation metrics.

Build a project that can be showcased in a portfolio.

🛠️ Technologies & Libraries
Common:
Python 3.x

numpy, pandas – Data manipulation

scikit-learn – Dataset loading, train-test splitting, standardization

matplotlib – Visualization

PyTorch Implementation (test_torch.ipynb):
torch – Core deep learning library

torch.nn – Neural network layers

torch.optim – Optimizers (Adam)

Manual training loop for full control

TensorFlow Implementation (test_tf.ipynb):
tensorflow / keras – Model building and training

Sequential API for rapid prototyping

Built-in .fit() training loop

📊 Dataset Information
Dataset Name: Breast Cancer Wisconsin Dataset
Samples: 569
Features: 30 numerical features (mean radius, texture, perimeter, etc.)
Target:

0 → Malignant

1 → Benign

📌 Dataset Feature Breakdown

🧠 Model Architecture
PyTorch & TensorFlow Shared Design
Input Layer: 30 neurons (one for each feature)

Hidden Layer(s): Dense layer(s) with ReLU activation

Output Layer: Single neuron with Sigmoid activation for binary classification

Diagram:

📂 Project Structure
bash
Copy
Edit
📦 Breast-Cancer-Classification
 ┣ 📜 test_torch.ipynb     # PyTorch version
 ┣ 📜 test_tf.ipynb        # TensorFlow/Keras version
 ┣ 📜 README.md            # Project documentation
 ┣ 🖼 dataset_breakdown.png
 ┣ 🖼 nn_architecture.png
 ┣ 🖼 training_flow.png
🚀 Getting Started
1️⃣ Install dependencies
bash
Copy
Edit
pip install torch tensorflow scikit-learn pandas matplotlib
2️⃣ Run Jupyter Notebook
bash
Copy
Edit
jupyter notebook test_torch.ipynb
jupyter notebook test_tf.ipynb
🧠 Learning Journey
What I Learned
Data Preprocessing: The importance of scaling data before training.

Model Architecture:

Input layer → Hidden layer(s) with ReLU → Output layer with Sigmoid.

Training Loops:

PyTorch: Fully manual loop with forward pass, loss calculation, backpropagation, and optimizer step.

TensorFlow: Simple .fit() method handles everything internally.

Loss Functions: BCELoss in PyTorch vs binary_crossentropy in Keras.

Evaluation: Using accuracy scores to measure model performance.

🔍 Training Process Flow

🔍 PyTorch vs TensorFlow – Key Takeaways
Feature	PyTorch	TensorFlow/Keras
Control	✅ High (manual loops)	⚠️ Lower (abstracted)
Ease of Use	⚠️ Steeper learning curve	✅ Beginner-friendly
Flexibility	✅ Customizable	⚠️ Requires subclassing for customization
Community	Large	Large

📈 Future Improvements
Add Dropout layers to prevent overfitting.

Implement cross-validation.

Visualize loss & accuracy curves for better analysis.

Extend to multi-class datasets.

🏷️ License
This project is licensed under the MIT License.
Feel free to fork, modify, and use it for learning purposes.
