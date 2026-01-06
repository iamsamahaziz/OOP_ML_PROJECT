# 🤖 Mini Scikit-Learn in Java

An object-oriented Machine Learning framework developed in Java, inspired by Scikit-Learn. This project implements regression and classification algorithms with data preprocessing capabilities.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Implemented Algorithms](#-implemented-algorithms)
- [Examples](#-examples)
- [Project Structure](#-project-structure)
- [Author](#-author)

## ✨ Features

- **Regression Algorithms**: Linear Regression, KNN Regression
- **Classification Algorithms**: KNN Classification
- **Data Preprocessing**: StandardScaler, MinMaxScaler
- **Evaluation Metrics**: R², Accuracy, MSE
- **Utilities**: Train/Test Split with random seed
- **OOP Architecture**: Design pattern with abstract `MLModel` class

## 🏗️ Architecture

The project follows a modular architecture based on object-oriented programming:

```
ml/
├── core/          # Base abstract classes
├── linear/        # Linear regression algorithms
├── knn/           # K-Nearest Neighbors algorithms
├── preprocessing/ # Normalization and standardization
├── metrics/       # Evaluation metrics
├── model_selection/ # Data splitting utilities
└── app/           # Main application and tests
```

## 🚀 Installation

### Prerequisites

- Java JDK 8 or higher
- A Java IDE (IntelliJ IDEA, Eclipse, VS Code) or command-line compiler

### Compilation

```bash
# Compile all files
javac -d bin app/*.java core/*.java linear/*.java knn/*.java preprocessing/*.java metrics/*.java model_selection/*.java

# Run the application
java -cp bin ml.app.Main
```

## 💻 Usage

### Basic Example - Linear Regression

```java
import ml.linear.LinearRegression;
import ml.model_selection.DataUtils;

// Prepare data
double[][] data = {
    {1.0, 2.1}, {2.0, 3.9}, {3.0, 6.1}, 
    {4.0, 7.9}, {5.0, 10.2}
};

// Split into train/test
DataUtils.SplitResult split = DataUtils.trainTestSplit(data, 0.2, 42);

// Create and train the model
LinearRegression model = new LinearRegression(0.01, 1000);
model.train(split.trainSet);

// Evaluate
double score = model.score(split.testSet);
System.out.println("R² Score: " + score);
```

### Example - KNN Classification

```java
import ml.knn.KNNClassification;

// Binary classification dataset
double[][] classData = {
    {1.0, 1.1, 0.0}, {1.2, 0.9, 0.0},
    {5.0, 5.1, 1.0}, {5.2, 4.9, 1.0}
};

// Create and train
KNNClassification model = new KNNClassification(3);
model.train(classData);

// Predict
double[] newPoint = {1.5, 1.0};
double prediction = model.predict(newPoint);
```

### Data Preprocessing

```java
import ml.preprocessing.StandardScaler;

StandardScaler scaler = new StandardScaler();
double[][] normalizedData = scaler.fitTransform(trainData);
double[][] normalizedTest = scaler.transform(testData);
```

## 🧮 Implemented Algorithms

### Regression

| Algorithm | Class | Hyperparameters |
|-----------|-------|-----------------|
| Linear Regression | `LinearRegression` | `learningRate`, `epochs` |
| KNN Regression | `KNNRegression` | `k` (number of neighbors) |

### Classification

| Algorithm | Class | Hyperparameters |
|-----------|-------|-----------------|
| KNN Classification | `KNNClassification` | `k` (number of neighbors) |

### Preprocessing

| Technique | Class | Description |
|-----------|-------|-------------|
| Standardization | `StandardScaler` | Z-score normalization (μ=0, σ=1) |
| Min-Max Scaling | `MinMaxScaler` | Scaling to [0, 1] range |

## 📊 Examples

The `Main.java` file contains 5 complete experiments:

1. **Learning Rate Impact** - Tests different learning rates (0.01, 0.005, 0.001)
2. **Epochs Impact** - Compares 500, 1000, 2000 iterations
3. **K Impact in KNN** - Evaluates K = 1, 3, 5, 7
4. **Preprocessing Impact** - Compares no preprocessing, MinMaxScaler, StandardScaler
5. **Test Ratio Impact** - Tests different train/test proportions (0.2, 0.3, 0.4)

### Running the Experiments

```bash
java -cp bin ml.app.Main
```

Expected output:
```
============================================================
    MINI SCIKIT-LEARN - COMPLETE EXPERIMENTS
============================================================

EXPERIMENT 1: Learning Rate Impact
--- Configuration: LR = 0.01 ---
Model: LinearRegression (ready)
LinearRegression model R² Score = 0.9876
...
```

## 📁 Project Structure

```
projet_poo/
│
├── app/
│   └── Main.java                    # Entry point and experiments
│
├── core/
│   └── MLModel.java                 # Base abstract class
│
├── linear/
│   └── LinearRegression.java        # Linear regression (Gradient Descent)
│
├── knn/
│   ├── KNNRegression.java          # K-Nearest Neighbors for regression
│   └── KNNClassification.java      # K-Nearest Neighbors for classification
│
├── preprocessing/
│   ├── Preprocessor.java           # Preprocessing interface
│   ├── StandardScaler.java         # Z-score standardization
│   └── MinMaxScaler.java           # Min-Max normalization
│
├── metrics/
│   └── Metrics.java                # Evaluation metrics (R², MSE, Accuracy)
│
├── model_selection/
│   └── DataUtils.java              # Train/Test Split
│
└── README.md                        # This file
```

## 🎓 OOP Concepts Used

- **Abstraction**: Abstract `MLModel` class defining the common interface
- **Inheritance**: All models inherit from `MLModel`
- **Polymorphism**: Methods `train()`, `predict()`, `score()` are overridden
- **Encapsulation**: Protected attributes and public methods
- **Interfaces**: `Preprocessor` for data transformers

## 🔬 Evaluation Metrics

- **R² (Coefficient of Determination)**: For regression (0 to 1, best = 1)
- **MSE (Mean Squared Error)**: Mean squared error
- **Accuracy**: Precision for classification (0 to 1, best = 1)

## 🛠️ Technologies

- **Language**: Java
- **Paradigm**: Object-Oriented Programming (OOP)
- **Inspiration**: Scikit-Learn (Python)

## 📝 License

This project is developed in an academic context.

## 👤 Author

**Samah AZIZ**
- GitHub: [@iamsamahaziz](https://github.com/iamsamahaziz)
- Project: OOP_ML_PROJECT

---

⭐ Feel free to star this project if you found it useful!
