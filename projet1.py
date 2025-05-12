import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Charger le modèle et le scaler
model = joblib.load('random_forest_aya.pkl')
scaler = joblib.load('scaler.pkl')

# Titre de l'application
st.set_page_config(page_title="Prédiction Cardiaque", layout="centered")
st.title("💓 Prédiction de Maladie Cardiaque")
st.markdown("Ce modèle estime la probabilité qu’un patient ait une **maladie cardiaque** à partir de données cliniques.")

# Image illustrative
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Heart_anterior_exterior_view.svg/1200px-Heart_anterior_exterior_view.svg.png", 
         width=300, caption="Illustration du cœur humain")

# Champs utilisateur dans la barre latérale
st.sidebar.header("📝 Informations du Patient")

age = st.sidebar.slider('Âge', 29, 77, 50)
sex = st.sidebar.radio("Sexe", ['Homme', 'Femme'])
sex = 1 if sex == 'Homme' else 0
cp = st.sidebar.selectbox("Type de douleur thoracique", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Tension au repos (mm Hg)", 90, 200, 120)
chol = st.sidebar.slider("Cholestérol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.radio("Glycémie à jeun > 120 mg/dl", ['Oui', 'Non'])
fbs = 1 if fbs == 'Oui' else 0
restecg = st.sidebar.selectbox("ECG au repos", [0, 1, 2])
thalach = st.sidebar.slider("Fréquence cardiaque max", 70, 210, 150)
exang = st.sidebar.radio("Angine d'effort", ['Oui', 'Non'])
exang = 1 if exang == 'Oui' else 0
oldpeak = st.sidebar.slider("Dépression ST", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Pente du ST", [0, 1, 2])
ca = st.sidebar.slider("Nb de vaisseaux colorés", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassémie", [1, 2, 3])

# Préparation des données
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal]])

# Mise à l'échelle
input_scaled = scaler.transform(input_data)

# Bouton de prédiction
if st.button("🔍 Prédire"):
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    st.subheader("🔎 Résultat de la Prédiction")
    if prediction == 1:
        st.markdown(f"❗ **Risque détecté** : Probabilité de maladie cardiaque **{proba:.2%}**")
        st.error("Consultez un médecin pour un diagnostic approfondi.")
    else:
        st.markdown(f"✅ **Pas de signe de maladie cardiaque** : Probabilité **{proba:.2%}**")
        st.success("Aucun symptôme apparent détecté.")

    # Récapitulatif des entrées
    st.subheader("📋 Résumé des Informations Saisies")
    features = ["Âge", "Sexe", "Douleur thoracique", "Tension", "Cholestérol", "FBS", "ECG", "Thalach",
                "Exang", "Oldpeak", "Slope", "CA", "Thal"]
    values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach,
              exang, oldpeak, slope, ca, thal]
    df = pd.DataFrame([values], columns=features)
    st.dataframe(df)

# Info modèle
st.markdown("---")
st.markdown("""
### ℹ️ À propos du modèle
Ce modèle est basé sur un algorithme **Random Forest Classifier** entraîné sur des données médicales.
Il a obtenu une **AUC de 1.00** pendant l'entraînement, ce qui indique une excellente performance sur l'ensemble de test.
""")
