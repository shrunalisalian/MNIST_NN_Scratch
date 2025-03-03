### 🧠 **Building a Neural Network from Scratch for MNIST Classification**  
This project implements a **Neural Network (NN) from scratch using NumPy** to classify handwritten digits from the **MNIST dataset**. Instead of using deep learning frameworks like TensorFlow or PyTorch, this project focuses on **understanding the fundamentals of neural networks**, including:  

✅ **Forward Propagation** – Computing activations and predictions  
✅ **Backpropagation** – Updating weights using gradient descent  
✅ **Activation Functions** – ReLU and Softmax for classification  
✅ **Loss Function** – Cross-Entropy for multi-class classification  
✅ **Weight Initialization & Regularization**  

By building this model from scratch, this project demonstrates **a strong understanding of deep learning principles**, essential for **ML & AI roles at top tech companies.**  

---

## 🎯 **Key Objectives**  
✔ **Implement a fully connected Neural Network from scratch**  
✔ **Train on the MNIST dataset without deep learning libraries**  
✔ **Understand the math behind backpropagation & gradient updates**  
✔ **Compare performance with modern deep learning frameworks**  

---

## 📊 **Dataset Overview: MNIST Handwritten Digits**  
The MNIST dataset consists of **28x28 grayscale images** of handwritten digits (0-9), used as a **benchmark for image classification models**.  

🔹 **60,000 training images**  
🔹 **10,000 test images**  
🔹 **Each image is a 784-dimensional vector (flattened 28x28 pixels)**  
🔹 **Labels range from 0 to 9 (multi-class classification)**  

✅ **Example: Visualizing MNIST Digits**  
```python
import matplotlib.pyplot as plt

plt.imshow(X_train[:, 0].reshape(28, 28), cmap="gray")
plt.title(f"Label: {Y_train[0]}")
plt.show()
```

---

## 🏗 **Neural Network Architecture**  
This NN has **two layers**:  

| **Layer** | **Units** | **Activation** | **Purpose** |
|-----------|----------|----------------|-------------|
| **Input Layer** | 784 | — | Pixels from MNIST images |
| **Hidden Layer** | 10 | **ReLU** | Captures patterns & features |
| **Output Layer** | 10 | **Softmax** | Multi-class classification |

✅ **Example: Forward Propagation Implementation**  
```python
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# Forward pass through the network
Z1 = np.dot(W1, X_train) + b1
A1 = relu(Z1)

Z2 = np.dot(W2, A1) + b2
A2 = softmax(Z2)  # Final probabilities
```

💡 **Why ReLU?**  
- Helps model **complex patterns** by introducing non-linearity.  
- Prevents **vanishing gradients**, unlike sigmoid/tanh.  

💡 **Why Softmax?**  
- Converts raw logits into **probabilities** for multi-class classification.  

---

## 🔥 **Training the Neural Network with Backpropagation**  
The model **updates weights iteratively** using **gradient descent** to minimize cross-entropy loss.  

✅ **Example: Backpropagation Implementation**  
```python
def compute_gradients(X, Y, A1, A2, W2):
    m = X.shape[1]  # Number of examples
    dZ2 = A2 - Y  # Gradient of output layer
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (A1 > 0)  # ReLU derivative
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2
```
💡 **Key Learnings from Backpropagation:**  
✔ **Gradient descent adjusts weights** by computing **partial derivatives of the loss function**.  
✔ **ReLU's derivative** is **1 for positive values, 0 for negatives** (solves vanishing gradient problem).  
✔ **Softmax + Cross-Entropy combination** ensures proper probability calibration.  

---

## 📈 **Model Performance & Accuracy**  
The neural network achieves:  
✔ **85% accuracy on training set**  
✔ **84% accuracy on validation set**  

✅ **Example: Evaluating the Model**  
```python
def compute_accuracy(predictions, labels):
    return np.mean(predictions == labels) * 100

predictions = np.argmax(A2, axis=0)
accuracy = compute_accuracy(predictions, Y_train)
print(f"Training Accuracy: {accuracy:.2f}%")
```

---

## 📊 **Comparison with Deep Learning Frameworks**  
How does a **scratch-built NN** compare with **TensorFlow/PyTorch implementations?**  

| **Model** | **Training Accuracy** | **Validation Accuracy** | **Notes** |
|-----------|----------------------|------------------------|-----------|
| **Scratch Neural Network (NumPy)** | **85%** | **84%** | Fully custom implementation |
| **TensorFlow/Keras NN** | **98%** | **97%** | Uses optimizations like Adam |
| **PyTorch CNN** | **99%** | **98%** | Uses convolutional layers |

💡 **Observations:**  
- **Scratch NN performs well** but lacks modern optimizations like **Adam optimizer & dropout**.  
- **Deep learning frameworks** provide **higher accuracy & efficiency** with minimal effort.  

---

## 🔮 **Future Enhancements**  
🔹 **Add Dropout Layers** – To prevent overfitting  
🔹 **Implement Mini-Batch Gradient Descent** – For efficient training  
🔹 **Extend to Convolutional Neural Networks (CNNs)** – To improve performance  
🔹 **Hyperparameter Optimization** – Tune learning rate, batch size, and weight initialization  

---

## 🎯 **Why This Project Stands Out for ML & AI Roles**  
✔ **Demonstrates Deep Learning Fundamentals** – No libraries, just NumPy!  
✔ **Hands-on Understanding of Forward & Backpropagation**  
✔ **Strong Math & Algorithmic Approach to ML**  
✔ **Foundation for Building Custom Deep Learning Architectures**  

---

## 🛠 **How to Run This Project**  
1️⃣ Clone the repo:  
   ```bash
   git clone https://github.com/shrunalisalian/nn-from-scratch-mnist.git
   ```
2️⃣ Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ Run the Jupyter Notebook:  
   ```bash
   jupyter notebook nn-scratch-mnist.ipynb
   ```

---

Referance: https://www.youtube.com/watch?v=w8yWXqWQYmU&ab_channel=SamsonZhang

---

## 📌 **Connect with Me**  
- **LinkedIn:** [Shrunali Salian](https://www.linkedin.com/in/shrunali-salian/)  
- **Portfolio:** [Your Portfolio Link](#)  
- **Email:** [Your Email](#)  
