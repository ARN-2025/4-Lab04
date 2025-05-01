# Introduction

Ce fichier n'est pas destiné à être push sur le repo, il est juste là pour que nous comprenons ce que nous faisons.

## Étape 1

Fichier `MLP_from_raw_data.ipynb`

### 🎯 Objectif global de l'étape 1

L’objectif est de classer des images de chiffres manuscrits (`MNIST`) en utilisant un réseau de neurones multicouches (`MLP`). On commence avec des données "brutes", c’est-à-dire les pixels directement, sans traitement préalable.

### 🔢 Pourquoi utiliser les pixels comme entrée ?

- Une image `MNIST` est composée de `28x28 = 784 pixels`, en niveaux de gris.
- Chaque pixel contient une information importante : c’est la représentation directe du chiffre.
- Utiliser les pixels bruts permet de tester la capacité du réseau à apprendre sans aide extérieure (pas d'extraction manuelle de caractéristiques).

### 🧠 Pourquoi un `MLP` (Multilayer Perceptron) ?

- Le `MLP` est une architecture simple et standard en deep learning, utile pour établir une base de comparaison.
- Il est dit “shallow” ici car il ne contient qu’une seule couche cachée.
- Il permet d’apprendre des relations non linéaires entre pixels et classes (0 à 9).

### 🏗️ Pourquoi tester plusieurs tailles de couches cachées (128, 256, 512) ?

- Une plus grande couche cachée permet de mieux approximer des fonctions complexes… mais :
    - demande plus de temps d'entraînement.
    - augmente le risque de surapprentissage (`overfitting`).
- Tester différentes tailles permet d’équilibrer capacité de généralisation et complexité.

### ⚙️ Pourquoi utiliser softmax en sortie ?

- Il s'agit d'une classification multiclasse (10 classes).
- La couche softmax retourne une probabilité par classe, ce qui permet d’interpréter la sortie comme une prédiction.

### 🧮 Pourquoi categorical_crossentropy comme fonction de perte ?

- C’est la fonction de perte adaptée aux problèmes de classification multiclasse avec one-hot encoding.
- Elle mesure l'écart entre la distribution prédite (softmax) et la vérité terrain (one-hot).

### 📈 Pourquoi visualiser accuracy et courbe de perte ?

Cela permet de :

- vérifier si le modèle apprend bien (val_accuracy ↑, val_loss ↓),
- détecter un overfitting (val_accuracy stagne ou diminue alors que train_accuracy ↑).

### 🔍 Pourquoi une matrice de confusion ?

- Elle permet d'analyser finement les erreurs du modèle.
- Elle montre quelles classes sont confondues entre elles, ce qui éclaire les limites du modèle.

## Étape 2

Fichier `MLP_from_HOG.ipynb`

### 🎯 Objectif global de l'étape 2

L’objectif est toujours de classer des chiffres manuscrits (`MNIST`) mais cette fois-ci **en extrayant des caractéristiques (features)** à l’aide de `HOG` avant d’entraîner un `MLP`.  
On cherche à comparer l’approche *brute* (pixels) avec une approche *basée sur des descripteurs visuels*.

---

### 🧪 Pourquoi utiliser `HOG` (Histogram of Oriented Gradients) ?

- HOG est une méthode classique d’extraction de caractéristiques visuelles.
- Elle décrit les **bords et contours** présents dans une image.
- Elle est plus **invariante aux variations locales** (bruit, petites déformations) que les pixels bruts.
- Elle permet au MLP d’apprendre sur des **informations plus compactes et discriminantes**.

---

### ⚙️ Pourquoi tester plusieurs `pix_per_cell` (4 et 7) ?

- Ce paramètre détermine la **résolution locale de détection des gradients** :
  - Une petite cellule (4x4) capte plus de détails.
  - Une grande cellule (7x7) donne une représentation plus globale.
- Comparer ces deux tailles permet de **mesurer l’impact de la granularité des descripteurs** sur les performances du réseau.

---

### 🔢 Pourquoi standardiser les features (`StandardScaler`) ?

- Les valeurs de HOG peuvent varier selon l’intensité des pixels et la taille des cellules.
- La standardisation (moyenne = 0, écart-type = 1) :
  - accélère l’apprentissage,
  - stabilise l’optimiseur,
  - empêche qu’une dimension domine les autres.

---

### 🧠 Pourquoi encore un `MLP` ?

- Le but est de comparer les mêmes types de modèles (**shallow MLP**) mais avec des **entrées différentes** (pixels bruts vs. HOG).
- Cela permet de voir si l’extraction de features améliore ou détériore la performance d’un réseau simple.

---

### 🏗️ Pourquoi tester plusieurs tailles de couches cachées (128, 256, 512) ?

- Même logique que pour l’étape 1 : tester la capacité d’approximation du réseau en fonction du nombre de neurones.

---

### 📊 Pourquoi comparer les courbes et matrices de confusion ?

- Comme à l’étape 1, cela permet de :
  - suivre la **convergence de l’apprentissage**,
  - identifier les **chiffres mal classés** malgré les features HOG,
  - **comparer objectivement avec les résultats obtenus avec les pixels bruts**.

---

### ⚖️ Conclusion

Cette étape permet de comprendre :

- L’intérêt d’utiliser des descripteurs (comme HOG) en entrée d’un réseau,
- Les impacts des choix de paramètres (cellule, neurones),
- Comment l’extraction de features modifie la performance d’un réseau simple.

## Étape 3
