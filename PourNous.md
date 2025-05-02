# Introduction

Ce fichier n'est pas destiné à être push sur le repo, il est juste là pour que nous comprenions ce que nous faisons.

## Étape 1 – MLP sur données brutes

Fichier : `MLP_from_raw_data.ipynb`  
**But** : Implémenter et comparer plusieurs réseaux MLP classiques (1 couche cachée), entraînés sur les pixels bruts du jeu de données MNIST.

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

## Étape 2 – MLP avec extraction HOG

Fichier : `MLP_from_HOG.ipynb`  
**But** : Comparer l’approche "pixels bruts" avec une approche où les images sont d'abord transformées en vecteurs de caractéristiques HOG, puis classées avec un MLP.

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

## Étape 3 – CNN

Fichier : `CNN.ipynb`  
**But** : Utiliser des réseaux convolutifs (CNN) pour améliorer les performances de classification sur MNIST, en explorant plusieurs architectures (filtres, dropout, etc.).

### 🎯 Objectif global de l'étape 3

L'objectif est de construire et d'entraîner plusieurs modèles CNN, en faisant varier les **paramètres structurels** comme le nombre de filtres, la taille des noyaux, la présence de Dropout, etc.  
Le but final est de comparer la **puissance des CNN** face aux MLP, et d'étudier l’impact de certains choix d’architecture.

---

### 🧠 Pourquoi un `CNN` (Convolutional Neural Network) ?

- Contrairement aux MLP, les CNN exploitent la **structure spatiale des images**.
- Ils utilisent des **filtres convolutifs** qui détectent des motifs (bords, textures, formes).
- Ils nécessitent **beaucoup moins de paramètres** pour des résultats souvent bien meilleurs.

---

### 🔁 Pourquoi tester plusieurs combinaisons (filtres, tailles, dropout) ?

- Chaque paramètre modifie la **capacité d’extraction des motifs visuels** :
  - **Plus de filtres** → plus de diversité de motifs détectés.
  - **Plus grand noyau (5x5)** → motifs plus larges (mais plus coûteux).
  - **Dropout** → réduction du surapprentissage.
- En testant **8 combinaisons**, on peut :
  - **mesurer l'impact de chaque paramètre**,
  - **identifier la meilleure configuration**,
  - conclure objectivement sur leur utilité.

---

### 📈 Pourquoi visualiser les courbes de validation ?

- Elles permettent de suivre **l'évolution de la précision sur les données non vues**.
- Une courbe stable qui monte suggère un bon apprentissage.
- On peut identifier si un modèle **sur-apprend** (écart train/val) ou **sous-apprend**.

---

### 🔍 Pourquoi une matrice de confusion finale ?

- Même logique que pour les étapes 1 et 2.
- Ici elle confirme que le CNN **fait moins d’erreurs de confusion** que les autres modèles.

---

### 🧪 Quel est le meilleur modèle et pourquoi ?

- Le modèle **F32_K5_DO_D128** est le meilleur (99.05% de précision test).
- Il utilise :
  - **32 filtres**
  - **noyaux 5x5**
  - **Dropout** (25% après conv, 50% après dense)
- Il montre que :
  - le **dropout aide clairement** à la généralisation,
  - un modèle modeste mais bien régularisé peut surpasser des architectures plus grandes.

---

### ⚖️ Comparaison avec les MLP précédents

| Modèle        | Accuracy test |
|---------------|---------------|
| MLP (raw)     | 98.26%        |
| MLP (HOG)     | 98.40%        |
| CNN (meilleur)| **99.05%**    |

Le CNN surpasse clairement les autres modèles, tout en conservant une structure assez simple.  
Cela confirme que les architectures convolutives sont **mieux adaptées à l’analyse d’images**, même simples comme MNIST.

## Étape 4 – CNN pour la détection de pneumonie

Fichier : `CNN_pneumonia.ipynb`  
**But** : Construire un CNN binaire (pneumonie / normal) à partir de radiographies pulmonaires en niveaux de gris.

### 🎯 Objectif global de l'étape 4
Détecter automatiquement la pneumonie sur des clichés thoraciques afin de démontrer la transférabilité des CNN à un **domaine médical plus complexe** que MNIST.

---

### 🧠 Pourquoi un `CNN` pour les radios pulmonaires ?
- Les radiographies contiennent des **patrons visuels locaux** (opacités, infiltrats) que les filtres convolutifs identifient mieux qu’un MLP.  
- Le CNN gère les **variations d’échelle et de position** grâce au pooling.  
- Le nombre de paramètres reste raisonnable par rapport à un réseau entièrement dense sur des images 150×150.

---

### 🏗️ Architecture retenue
1. **Conv 32 (3×3) → ReLU → MaxPool**  
2. **Conv 64 (3×3) → ReLU → MaxPool**  
3. **Conv 128 (3×3) → ReLU → MaxPool**  
4. **Dropout 0.5**  
5. **Flatten → Dense 128 (ReLU) → Dense 1 (sigmoïde)**  

*Hyper‑paramètres* : Adam, `binary_crossentropy`, batch 32, 10 époques (early‑stopping activé).

---

### ⚙️ Pourquoi ces choix ?
- **Images 150×150 / niveaux de gris** : compromis entre définition et mémoire GPU.  
- **Sigmoïde + binary_crossentropy** : adapté à une classification **binaire**.  
- **Dropout 0.5** : essentiel pour **réduire le surapprentissage** sur un dataset limité.  
- **Recall priorisé** : rater une pneumonie est plus grave qu’un faux positif.

---

### 📈 Résultats principaux
| Classe        | Précision | Rappel | F1‑score |
|---------------|-----------|--------|----------|
| Pneumonia     | 0.81      | **0.99** | 0.88     |
| Normal        | **0.95**  | 0.58   | 0.73     |

- **Accuracy globale** : 83.8 % (sur le jeu de test).  
- **Matrice de confusion** : très peu de faux négatifs, mais un nombre notable de faux positifs (normaux ➜ pneumonie).

---

### 🔍 Analyse des performances
- **Sensibilité élevée** (rappel 0.99) : le modèle repère quasi toutes les pneumonies, ce qui est primordial en clinique.  
- **Spécificité limitée** : beaucoup de normaux incorrectement classés ➜ besoin de régularisation ou de rééquilibrage.  
- **Déséquilibre de classes** : le dataset contient ~3 × plus d’images de pneumonie que de normales, d’où le biais.

---

### 🛠️ Pistes d’amélioration
- **Data augmentation ciblée** (rotations légères, translations) pour enrichir la classe *Normal*.  
- **Rééchantillonnage ou pondération des classes** dans la fonction de perte.  
- **Batch Normalization** entre convolutions pour stabiliser l’apprentissage.  
- **Entraînement plus long + early‑stopping** afin d’atteindre un meilleur point de validation.

---

### ⚖️ Conclusion
Le CNN détecte efficacement la pneumonie (très peu de faux négatifs) mais doit réduire les faux positifs avant toute application clinique réelle. Des techniques de ré‑équilibrage et de régularisation devraient permettre d’améliorer la **spécificité** sans sacrifier la **sensibilité**.
