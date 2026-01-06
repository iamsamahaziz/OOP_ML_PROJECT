# 🤖 Mini Scikit-Learn en Java

Un framework de Machine Learning orienté objet développé en Java, inspiré de Scikit-Learn. Ce projet implémente des algorithmes de régression et de classification avec des fonctionnalités de prétraitement des données.

## 📋 Table des matières

- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Algorithmes implémentés](#-algorithmes-implémentés)
- [Exemples](#-exemples)
- [Structure du projet](#-structure-du-projet)
- [Auteur](#-auteur)

## ✨ Fonctionnalités

- **Algorithmes de régression** : Régression linéaire, KNN Régression
- **Algorithmes de classification** : KNN Classification
- **Prétraitement des données** : StandardScaler, MinMaxScaler
- **Métriques d'évaluation** : R², Accuracy, MSE
- **Utilitaires** : Train/Test Split avec seed aléatoire
- **Architecture POO** : Design pattern avec classe abstraite `MLModel`

## 🏗️ Architecture

Le projet suit une architecture modulaire basée sur la programmation orientée objet :

```
ml/
├── core/          # Classes abstraites de base
├── linear/        # Algorithmes de régression linéaire
├── knn/           # Algorithmes K-Nearest Neighbors
├── preprocessing/ # Normalisation et standardisation
├── metrics/       # Métriques d'évaluation
├── model_selection/ # Utilitaires de séparation des données
└── app/           # Application principale et tests
```

## 🚀 Installation

### Prérequis

- Java JDK 8 ou supérieur
- Un IDE Java (IntelliJ IDEA, Eclipse, VS Code) ou un compilateur en ligne de commande

### Compilation

```bash
# Compiler tous les fichiers
javac -d bin app/*.java core/*.java linear/*.java knn/*.java preprocessing/*.java metrics/*.java model_selection/*.java

# Exécuter l'application
java -cp bin ml.app.Main
```

## 💻 Utilisation

### Exemple basique - Régression linéaire

```java
import ml.linear.LinearRegression;
import ml.model_selection.DataUtils;

// Préparer les données
double[][] data = {
    {1.0, 2.1}, {2.0, 3.9}, {3.0, 6.1}, 
    {4.0, 7.9}, {5.0, 10.2}
};

// Séparer en train/test
DataUtils.SplitResult split = DataUtils.trainTestSplit(data, 0.2, 42);

// Créer et entraîner le modèle
LinearRegression model = new LinearRegression(0.01, 1000);
model.train(split.trainSet);

// Évaluer
double score = model.score(split.testSet);
System.out.println("Score R² : " + score);
```

### Exemple - KNN Classification

```java
import ml.knn.KNNClassification;

// Dataset de classification binaire
double[][] classData = {
    {1.0, 1.1, 0.0}, {1.2, 0.9, 0.0},
    {5.0, 5.1, 1.0}, {5.2, 4.9, 1.0}
};

// Créer et entraîner
KNNClassification model = new KNNClassification(3);
model.train(classData);

// Prédire
double[] newPoint = {1.5, 1.0};
double prediction = model.predict(newPoint);
```

### Prétraitement des données

```java
import ml.preprocessing.StandardScaler;

StandardScaler scaler = new StandardScaler();
double[][] normalizedData = scaler.fitTransform(trainData);
double[][] normalizedTest = scaler.transform(testData);
```

## 🧮 Algorithmes implémentés

### Régression

| Algorithme | Classe | Hyperparamètres |
|------------|--------|-----------------|
| Régression Linéaire | `LinearRegression` | `learningRate`, `epochs` |
| KNN Régression | `KNNRegression` | `k` (nombre de voisins) |

### Classification

| Algorithme | Classe | Hyperparamètres |
|------------|--------|-----------------|
| KNN Classification | `KNNClassification` | `k` (nombre de voisins) |

### Prétraitement

| Technique | Classe | Description |
|-----------|--------|-------------|
| Standardisation | `StandardScaler` | Normalisation Z-score (μ=0, σ=1) |
| Min-Max Scaling | `MinMaxScaler` | Mise à l'échelle [0, 1] |

## 📊 Exemples

Le fichier `Main.java` contient 5 expériences complètes :

1. **Impact du Learning Rate** - Teste différents taux d'apprentissage (0.01, 0.005, 0.001)
2. **Impact du nombre d'Epochs** - Compare 500, 1000, 2000 itérations
3. **Impact de K dans KNN** - Évalue K = 1, 3, 5, 7
4. **Impact du prétraitement** - Compare sans prétraitement, MinMaxScaler, StandardScaler
5. **Impact du Test Ratio** - Teste différentes proportions train/test (0.2, 0.3, 0.4)

### Exécution des expériences

```bash
java -cp bin ml.app.Main
```

Sortie attendue :
```
============================================================
    MINI SCIKIT-LEARN - EXPÉRIMENTATIONS COMPLÈTES
============================================================

EXPÉRIENCE 1 : Impact du Learning Rate
--- Configuration : LR = 0.01 ---
Modèle : LinearRegression (prêt)
Score (R2) du modèle LinearRegression = 0.9876
...
```

## 📁 Structure du projet

```
projet_poo/
│
├── app/
│   └── Main.java                    # Point d'entrée et expériences
│
├── core/
│   └── MLModel.java                 # Classe abstraite de base
│
├── linear/
│   └── LinearRegression.java        # Régression linéaire (Gradient Descent)
│
├── knn/
│   ├── KNNRegression.java          # K-Nearest Neighbors pour régression
│   └── KNNClassification.java      # K-Nearest Neighbors pour classification
│
├── preprocessing/
│   ├── Preprocessor.java           # Interface de prétraitement
│   ├── StandardScaler.java         # Standardisation Z-score
│   └── MinMaxScaler.java           # Normalisation Min-Max
│
├── metrics/
│   └── Metrics.java                # Métriques d'évaluation (R², MSE, Accuracy)
│
├── model_selection/
│   └── DataUtils.java              # Train/Test Split
│
└── README.md                        # Ce fichier
```

## 🎓 Concepts POO utilisés

- **Abstraction** : Classe abstraite `MLModel` définissant l'interface commune
- **Héritage** : Tous les modèles héritent de `MLModel`
- **Polymorphisme** : Méthodes `train()`, `predict()`, `score()` redéfinies
- **Encapsulation** : Attributs protégés et méthodes publiques
- **Interfaces** : `Preprocessor` pour les transformateurs de données

## 🔬 Métriques d'évaluation

- **R² (Coefficient de détermination)** : Pour la régression (0 à 1, meilleur = 1)
- **MSE (Mean Squared Error)** : Erreur quadratique moyenne
- **Accuracy** : Précision pour la classification (0 à 1, meilleur = 1)

## 🛠️ Technologies

- **Langage** : Java
- **Paradigme** : Programmation Orientée Objet (POO)
- **Inspiration** : Scikit-Learn (Python)

## 📝 Licence

Ce projet est développé dans un cadre académique.

## 👤 Auteur

**Sara Mahaziz**
- GitHub: [@iamsamahaziz](https://github.com/iamsamahaziz)
- Projet: OOP_ML_PROJECT

---

⭐ N'hésitez pas à mettre une étoile si ce projet vous a été utile !

