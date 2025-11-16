import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

# === Configuration ===
IMAGE_SIZE = (100, 100)
NUM_SIFT_FEATURES = 50

# =========================== Fonctions utilitaires ===========================
@st.cache_data(max_entries=50, ttl=3600) 
# Chargement des images
def charger_donnees(base_path):
    images, labels = [], []
    for file in os.listdir(base_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            path = os.path.join(base_path, file)
            img = cv2.imread(path)
            img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            person_id = int(file.split("-")[0])
            images.append(img)
            labels.append(person_id - 1)
    return np.array(images), np.array(labels)


# Prétraitement des images
def preprocess_image(img_input):
    # Si l'input est un chemin de fichier
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    # Si l'input est déjà une image numpy array
    else:
        img = img_input.copy()
    if img is None:
        return None
        
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Détection du visage
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    visages = cascade.detectMultiScale(img_gris, 1.1, 5)
    if len(visages) == 0:
        return None
    x,y,w,h = visages[0]
    visage = img_gris[y:y+h, x:x+w]
    
    # Redimensionnement et normalisation
    visage = cv2.resize(visage, IMAGE_SIZE)
    visage = cv2.normalize(visage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    return visage




# LBP
@st.cache_data
def apply_lbp(image):
    # Convertir en niveaux de gris si nécessaire
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    lbp = np.zeros_like(gray)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            center = gray[i, j]
            binary = (gray[i-1:i+2, j-1:j+2] >= center).astype(int)
            binary[1,1] = 0
            weights = [1, 2, 4, 8, 0, 16, 32, 64, 128]
            lbp[i, j] = np.dot(binary.flatten(), weights)
    return lbp

def visualize_sift_keypoints(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)
    
    # Dessiner les keypoints avec orientation et échelle
    img_with_keypoints = cv2.drawKeypoints(
        image, 
        keypoints, 
        None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)  
    )
    return img_with_keypoints

@st.cache_data
def extract_sift_histogram(image, n_features=NUM_SIFT_FEATURES):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
        
    # Gestion des descripteurs manquants
    if descriptors is None:
        descriptors = np.zeros((n_features, 128))
    elif descriptors.shape[0] < n_features:
        descriptors = np.vstack([descriptors, np.zeros((n_features - descriptors.shape[0], 128))])
    elif descriptors.shape[0] > n_features:
        descriptors = descriptors[:n_features]
    
    return descriptors.flatten()

# Combinaison LBP + SIFT
@st.cache_data
def extract_features(images):
    features = []
    for img in images:
        lbp = apply_lbp(img).flatten()
        sift_hist = extract_sift_histogram(img)
        combined = np.concatenate([lbp, sift_hist])
        features.append(combined)
    return np.array(features)

# fonctions de visualisation
def visualize_lbp_sift(original, lbp, sift_keypoints_img):    
    # Convertir LBP en couleur pour superposition
    lbp_color = cv2.cvtColor(lbp.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # Fusionner les visualisations
    combined = cv2.addWeighted(lbp_color, 0.7, sift_keypoints_img, 0.3, 0)
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    # Image originale couleur
    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Originale")
    ax[0].axis('off')
    
    # Carte LBP
    ax[1].imshow(lbp, cmap='gray')
    ax[1].set_title("Texture LBP")
    ax[1].axis('off')
    
    # Points clés SIFT
    ax[2].imshow(cv2.cvtColor(sift_keypoints_img, cv2.COLOR_BGR2RGB))
    ax[2].set_title("Points Clés SIFT")
    ax[2].axis('off')
    
    # Combinaison
    ax[3].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    ax[3].set_title("Fusion LBP + SIFT")
    ax[3].axis('off')
    
    return fig


def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        # Couche de convolution pour les motifs LBP
        Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        
        # Feature extraction profond
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        # Classification
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

@st.cache_data
def train_model(images, labels):
    lbp_images = np.array([apply_lbp(img) for img in images])
    lbp_norm = lbp_images / 255.0
    lbp_norm = lbp_norm.reshape(-1, *IMAGE_SIZE, 1)
    y_cat = to_categorical(labels)
    X_train, X_test, y_train, y_test = train_test_split(lbp_norm, y_cat, test_size=0.2, stratify=labels)

    model = build_cnn_model((100, 100, 1), y_cat.shape[1])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    return model, history, X_test, y_test, y_cat

def get_similar_images(pred_label, all_labels, all_images):
    return all_images[all_labels == pred_label]







# =================== Interface Streamlit ======================
st.set_page_config(page_title="Reconnaissance Faciale IA", layout="wide", page_icon="🧠")

st.markdown("""
<style>
    .main {
        background-color: #1e1e1e;
        color: white;
    }
    h1, h2, h3, .stMarkdown, .stTextInput, .stButton, .stFileUploader, .stSelectbox {
        color: white !important;
    }
    .stButton > button {
        background-color: #2a9d8f;
        color: white;
        border-radius: 10px;
        padding: 0.5em 2em;
    }
    .stTextInput > div > input {
        background-color: #2c2c2c;
        color: white;
    }
    .stFileUploader {
        background-color: #2c2c2c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Reconnaissance Faciale avec LBP, SIFT et CNN")

st.markdown("""
Cette application vous permet de :
- 📂 Charger une image de personne,
- 🔎 Appliquer LBP et SIFT,
- 🧠 Entraîner un CNN pour la reconnaissance faciale,
- 📊 Évaluer le modèle avec des métriques et une matrice de confusion.
- 📷 Prédire une image et afficher la personne correspondante,
""")


with st.spinner("Chargement des données..."):
    chemin_base = "images"
    images, labels = charger_donnees(chemin_base)
    
    st.header("Taille de la base des données")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"Personnes totales: {len(np.unique(labels))}")

    with col2:
        st.info(f"Images totales: {len(images)}")
    with col3:
        st.write("")
        
        
st.header("1.📊 Visualisation de la phase de prétraitement")
fichier_upload = st.file_uploader("Choisir une image depuis votre ordinateur", type=['jpg', 'jpeg', 'png'])

if fichier_upload is not None:

    fichier_bytes = np.asarray(bytearray(fichier_upload.read()), dtype=np.uint8)
    img_originale = cv2.imdecode(fichier_bytes, cv2.IMREAD_COLOR)
    
    # Prétraitement
    img_traitee = preprocess_image(img_originale)
    
    # Affichage 
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_originale, channels="BGR", caption="Image Originale")
    
    with col2:
        if img_traitee is not None:
            st.image(img_traitee, caption="Image Traitée")

        else:
            st.error("Aucun visage détecté dans l'image!")
else:
    st.info(" Veuillez uploader une image pour le prétraiter")
    
    
    
    
st.header("2. 🔎 Extraction des caractèristiques avec LBP et SIFT")
with st.spinner("⏳ Extraction en cours..."):
    images, labels = charger_donnees(chemin_base)
    lbp_sample = apply_lbp(images[0])
    gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    sift_sample = extract_sift_histogram(gray)
    fig = visualize_lbp_sift(images[0], lbp_sample, sift_keypoints_img=visualize_sift_keypoints(images[0]))
    st.pyplot(fig)
    
    
    
    
    
st.header("3. 🤖 Modèle CNN")
with st.spinner("Construction du modèle en cours..."):
    model, history, X_test, y_test, y_cat = train_model(images, labels)
    st.success("Modèle construit avec succès!")

st.subheader("📈 Courbes d'entraînement")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(history.history['accuracy'], label='Train')
ax[0].plot(history.history['val_accuracy'], label='Val')
ax[0].set_title("Accuracy")
ax[0].legend()
ax[1].plot(history.history['loss'], label='Train')
ax[1].plot(history.history['val_loss'], label='Val')
ax[1].set_title("Loss")
ax[1].legend()
st.pyplot(fig)

st.header("4. 🧪 Évaluation du modèle")
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred_labels)
prec = precision_score(y_true, y_pred_labels, average='weighted')
rec = recall_score(y_true, y_pred_labels, average='weighted')
f1 = f1_score(y_true, y_pred_labels, average='weighted')

st.markdown(f"""
- ✅ *Exactitude :* {acc*100:.2f}%
- 🎯 *Precision :* {prec*100:.2f}%
- 🔁 *Rappel :* {rec*100:.2f}%
- 📐 *F1 Score :* {f1*100:.2f}%
""")

# Matrice de confusion
st.subheader("📌 Matrice de confusion")
with st.spinner("⏳ Création de matrice de confusion..."):
    cm = confusion_matrix(y_true, y_pred_labels)
    fig_cm, ax_cm = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax_cm)
    ax_cm.set_xlabel("Prédit")
    ax_cm.set_ylabel("Réel")
    ax_cm.set_title("📌 Matrice de confusion")
    st.pyplot(fig_cm)

st.header("5. 📷 Prédiction personnalisée")
uploaded_file = st.file_uploader("📂 Téléchargez une image à prédire", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("⏳ Recherche de la personne sélectionnée..."):

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        with st.spinner("Analyse de l'image..."):
            img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            
            if img_gris is None:
                st.error("Aucun visage détecté - Veuillez uploader une image claire")
            else:
                # Transformation pour le modèle
                lbp = apply_lbp(img_gris)
                lbp_norm = lbp.astype('float32')/255
                lbp_input = lbp_norm.reshape(1, *IMAGE_SIZE, 1)
                
                # Prédiction
                predictions = model.predict(lbp_input)
                pred_class = np.argmax(predictions)
                confidence = np.max(predictions) * 100
                
                # Affichage des résultats
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, channels="BGR", caption="Image originale")
                    
                with col2:
                    st.markdown(f"### Résultat de la prédiction : Personne ID {pred_class + 1}")
                    st.markdown(f"**Confiance :** {confidence:.2f}%")
                    
                # Récupération des images similaires
                similar_imgs = get_similar_images(pred_class, labels, images)
                
                # Affichage dans une grille responsive
                st.subheader(f"Les images de référence pour la personne ID {pred_class + 1}")
                
                cols = st.columns(4)  # 4 colonnes
                for i, img in enumerate(similar_imgs[:14]):
                    with cols[i % 4]:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                            caption=f"Image {i+1}", 
                            use_column_width=True)                                 
else:
    st.info("Veuillez importer une image pour effectuer une prédiction.")