# 🧠 Reconnaissance Faciale avec LBP, SIFT et CNN

Ce projet propose une **application de reconnaissance faciale** développée avec **Python**, **OpenCV**, **TensorFlow** et **Streamlit**.  
L’objectif est de combiner différentes approches **d’extraction de caractéristiques (LBP + SIFT)** avec un **réseau de neurones convolutif (CNN)** pour créer un système performant, pédagogique et interactif.

**Lien de la base de donnés :** https://fei.edu.br/~cet/facedatabase.html

---

## 🚀 Fonctionnalités

### 🔎 **1. Prétraitement des images**
- Détection automatique du visage (Haar Cascade)
- Conversion en niveaux de gris
- Normalisation & redimensionnement
- Affichage avant/après

### 🧩 **2. Extraction de caractéristiques**
- **LBP (Local Binary Pattern)** : texture du visage  
- **SIFT (Scale-Invariant Feature Transform)** : points clés robustes  
- Visualisation :  
  - Image LBP  
  - Points clés SIFT  
  - Fusion LBP + SIFT  

### 🤖 **3. Modèle CNN**

**Architecture du modèle CNN**
```
- Conv2D(16, 3x3) + ReLU  
- MaxPooling2D(2x2)  
- Conv2D(32, 3x3) + ReLU  
- MaxPooling2D(2x2)  
- Dense(64) + Dropout(0.5)  
- Dense(N classes) + Softmax  
```

- Entraînement sur cartes LBP  
- Courbes **Accuracy** & **Loss**
- Résultats :  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - Matrice de confusion  

### 📷 **4. Prédiction personnalisée**
- Upload d’une nouvelle image  
- Extraction LBP + prédiction CNN  
- Score de confiance  
- Affichage des images similaires de la même personne  

### 🌐 **5. Interface utilisateur (Streamlit)**
- Interface moderne & responsive  
- Visualisation complète du pipeline  
- Mode web interactif  

---

## 📂 Structure du projet

```
📦 Reconnaissance-Faciale
│
├── images/                     # Dataset (images de la base FEI)
├── app.py                      # Application Streamlit principale
├── requirements.txt            # Liste des dépendances
└── README.md                   # Documentation du projet
```

---

## 🛠️ Installation & Exécution

### 1️⃣ **Cloner le projet**
```bash
git clone https://github.com/ritejth/facial_recognition_project.git
cd facial_recognition_project
```

### 2️⃣ **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```

### 3️⃣ **Installer les dépendances**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Lancer l'application**
```bash
streamlit run app.py
```

---

## 📌 Points forts du projet

✔️ Combinaison de méthodes classiques (LBP, SIFT) + deep learning  
✔️ Pipeline complet prêt à l’emploi  
✔️ Interface web interactive  
✔️ Visualisation pédagogique  
✔️ Code optimisé et structuré  

---

## 👤 Auteur

**Ritej Touhami**  
Étudiante en Master Professionnel en Ingénierie des Systèmes d’Information & Data Science.
📧 ritejtouhami@gmail.com
🔗 LinkedIn
---

## 📄 Licence

Ce projet est disponible sous licence **MIT**.
