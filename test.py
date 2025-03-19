import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Configuration Streamlit
st.title("Analyse de Données - Diabète")
st.write("Cette application permet d'explorer et d'analyser un dataset sur le diabète.")

# Charger le dataset
file_path = 'C:\\Users\\User\\Desktop\\PFE\\diabetes_1_cleaned.csv'
dataset = pd.read_csv(file_path)

# Afficher les informations du dataset
st.header("Aperçu du Dataset")
st.dataframe(dataset)

# Dimensions du dataset
st.write("Dimensions du dataset : ", dataset.shape)

# Information sur les colonnes
st.write("**Informations sur les colonnes :**")
buffer = []
dataset.info(buf=buffer.append)
st.text("".join(buffer))

# Description statistique
st.write("**Statistiques descriptives :**")
st.write(dataset.describe())

# Vérifier les valeurs manquantes
st.write("**Valeurs manquantes :**")
st.write(dataset.isnull().sum())

# Affichage du heatmap des valeurs manquantes
st.write("**Heatmap des valeurs manquantes :**")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(dataset.isna(), cbar=False, ax=ax)
st.pyplot(fig)

# Affichage des colonnes
st.write("**Colonnes du dataset :**")
st.write(dataset.columns.tolist())

# Variables X et y
X = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']

# Afficher X et y
st.write("**Variables explicatives (X) :**")
st.write(X.head())
st.write("**Variable cible (y) :**")
st.write(y.head())

# Diviser les données
st.write("**Division des données en ensemble d'entraînement et de test :**")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write("Distribution des classes dans l'ensemble d'entraînement :")
st.write(y_train.value_counts())
st.write("Distribution des classes dans l'ensemble de test :")
st.write(y_test.value_counts())

# Application de SMOTE
st.write("**Application de SMOTE pour équilibrer les classes :**")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
st.write("Distribution des classes après SMOTE :")
st.write(y_train_smote.value_counts())

# Analyse des colonnes
st.write("**Analyse des colonnes spécifiques :**")
columns_to_analyze = [
    'Pregnancies', 'Glucose', 'BloodPressure', 
    'SkinThickness', 'Insulin', 'BMI', 
    'DiabetesPedigreeFunction', 'Age'
]
for col in columns_to_analyze:
    st.write(f"Valeurs uniques dans la colonne **{col}** :")
    st.write(dataset[col].value_counts())
