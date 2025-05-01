# Report ARN Lab04

@authors: Parisod Nathan & Lestiboudois Maxime </br>
@Date: 07.05.2025

## Step 1: MLP with Raw Data

### 1. Chosen Architecture

The input of the network consists of flattened `MNIST` images (28x28 = 784 pixels).  
The output is a 10-dimensional vector representing the digit classes (0–9).

We tested three architectures with one hidden layer:

- MLP with 128 hidden neurons
- MLP with 256 hidden neurons
- MLP with 512 hidden neurons

The best validation accuracy was obtained with **512 hidden neurons**.

**Final architecture:**

- Input layer: 784 neurons
- Hidden layer: 512 neurons (ReLU activation)
- Output layer: 10 neurons (Softmax activation)

---

### 2. Number of Parameters

For the final model (784 → 512 → 10):

- **Input to hidden layer:**
  - Weights: 784 × 512 = 401,408
  - Biases: 512
- **Hidden to output layer:**
  - Weights: 512 × 10 = 5,120
  - Biases: 10
- **Total parameters: 407,050**

---

### 3. Accuracy and Loss Evolution

The training was performed over 10 epochs, using a batch size of 128 and 10% of training data used for validation.

**Validation accuracy per epoch** showed continuous improvement for all models.  
The model with 512 neurons reached over **98.4% validation accuracy**.

Loss curves confirmed proper convergence without signs of overfitting.

---

### 4. Final Confusion Matrix

The final model was evaluated on the test set. It achieved a test accuracy of **98.26%**.

Confusion matrix analysis reveals:

- Most digits are very well classified.
- Common confusions include:
  - **4 and 9**
  - **5 and 3**
  - **7 and 2**

These errors are consistent with human visual ambiguities.

---

### 5. Comments

This first experiment shows that a shallow MLP trained on raw pixels can already achieve excellent performance on `MNIST`.  
However, certain digit pairs remain challenging and could benefit from feature extraction or deeper architectures.

## Step 2: TODO
