import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ----------------------------
# 1. Configuration de la Page
# ----------------------------
st.set_page_config(page_title="📱 Smartphone Prediction", layout="wide")

# ----------------------------
# 2. Chargement et Nettoyage (Tâche 3)
# ----------------------------
@st.cache_data
def load_and_clean():
    # Chargement du dataset
    df = pd.read_csv("../data/smartphones.csv")
    
    # Traitement des valeurs manquantes (Tâche 3a & 3c)
    df['RAM_GB'] = df['RAM_GB'].fillna(df['RAM_GB'].median())
    df['Storage_GB'] = df['Storage_GB'].fillna(df['Storage_GB'].median())
    df['Screen_Size_Inch'] = df['Screen_Size_Inch'].fillna(df['Screen_Size_Inch'].mean())
    df['Battery_mAh'] = df['Battery_mAh'].fillna(df['Battery_mAh'].median())
    
    # Suppression des anomalies extrêmes
    df = df[df['Price_DH'] < 50000] 
    
    return df

df = load_and_clean()

# ----------------------------
# 3. Préparation des données & Encodage (Tâche 5)
# ----------------------------
# Encodage One-Hot des marques
df_encoded = pd.get_dummies(df, columns=['Brand'], drop_first=True)
numerical_features = ['RAM_GB', 'Storage_GB', 'Screen_Size_Inch', 'Battery_mAh']

# Normalisation (StandardScaler)
scaler = StandardScaler()
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# Transformation Logarithmique de la cible (ESSENTIEL pour la précision)
# On crée la cible y en log pour stabiliser la variance
y_log = np.log1p(df['Price_DH'])

# Préparation de X (on retire les colonnes inutiles)
X = df_encoded.drop(columns=['Price_DH', 'Model'], errors='ignore')

# ----------------------------
# 4. Modélisation (Tâche 8)
# ----------------------------
# On limite la profondeur pour éviter le sur-apprentissage (Overfitting)
dt = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=42)
dt.fit(X, y_log)

# ----------------------------
# 5. Interface Utilisateur
# ----------------------------
st.sidebar.header("Navigation du Projet")
menu = st.sidebar.radio("Sélectionnez une étape :", 
                        ["Exploration & Nettoyage", 
                         "Réduction Dimensionnelle", 
                         "Analyse de l'Arbre", 
                         "Prédiction en Direct"])

st.title("📱 Analyse des Prix Des Smartphones")

# --- SECTION 1: Exploration ---
if menu == "Exploration & Nettoyage":
    st.header("🔍 Visualisation & Nettoyage (Tâches 2 & 3)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribution des Prix**")
        fig, ax = plt.subplots()
        sns.histplot(df["Price_DH"], kde=True, color="purple", ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("**Corrélation des caractéristiques**")
        fig2, ax2 = plt.subplots()
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

# --- SECTION 2: PCA/t-SNE ---
elif menu == "Réduction Dimensionnelle":
    st.header("🧩 Réduction de la Dimensionnalité (Tâche 6)")
    c1, c2 = st.columns(2)
    pca_res = PCA(n_components=2).fit_transform(X)
    tsne_res = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(X)
    with c1:
        st.write("**ACP (Analyse en Composantes Principales)**")
        fig3, ax3 = plt.subplots()
        ax3.scatter(pca_res[:,0], pca_res[:,1], c='orange', edgecolor='k', alpha=0.7)
        st.pyplot(fig3)
    with c2:
        st.write("**t-SNE (Visualisation non-linéaire)**")
        fig4, ax4 = plt.subplots()
        ax4.scatter(tsne_res[:,0], tsne_res[:,1], c='dodgerblue', edgecolor='k', alpha=0.7)
        st.pyplot(fig4)

# --- SECTION 3: Decision Tree ---
elif menu == "Analyse de l'Arbre":
    st.header("🌳 Modélisation (Tâche 8)")
    importance = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
    st.subheader("Identification des caractéristiques pertinentes")
    fig_imp, ax_imp = plt.subplots()
    sns.barplot(x=importance.values, y=importance.index, palette="viridis", ax=ax_imp)
    st.pyplot(fig_imp)
    st.divider()
    st.subheader("Visualisation de l'Arbre de Décision")
    fig5, ax5 = plt.subplots(figsize=(25, 12))
    plot_tree(dt, feature_names=list(X.columns), filled=True, rounded=True, fontsize=10)
    st.pyplot(fig5)

# --- SECTION 4: Prediction ---
elif menu == "Prédiction en Direct":
    st.header("🔮 Test de l'Application (Tâche 7)")
    col_input, col_res = st.columns([1, 1])
    
    with col_input:
        brand_choice = st.selectbox("Choisir la marque", df['Brand'].unique())
        ram_val = st.number_input("RAM (GB)", 2, 64, 8)
        storage_val = st.number_input("Stockage (GB)", 16, 1024, 128)
        screen_val = st.slider("Taille écran (Pouces)", 4.0, 8.0, 6.1)
        battery_val = st.slider("Batterie (mAh)", 1000, 10000, 5000)

    # LOGIQUE DE PRÉDICTION CORRIGÉE
    # 1. Créer le DataFrame numérique
    input_data = pd.DataFrame([[ram_val, storage_val, screen_val, battery_val]], 
                              columns=numerical_features)
    
    # 2. Mise à l'échelle (Scaling)
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])
    
    # 3. Gérer l'encodage des marques (One-Hot)
    for col in X.columns:
        if col.startswith("Brand_"):
            input_data[col] = 1 if col == f"Brand_{brand_choice}" else 0
            
    # 4. Aligner les colonnes avec le modèle X
    input_data = input_data[X.columns]
    
    # 5. Prédire et convertir depuis le Logarithme
    prediction_log = dt.predict(input_data)[0]
    prediction = np.expm1(prediction_log) # Conversion Log -> DH

    with col_res:
        st.markdown(f"""
            <div style="background-color:#f0f2f6; padding:40px; border-radius:15px; border: 2px solid #4B0082; text-align:center; margin-top:50px;">
                <h3 style="color:#4B0082;">Résultat de la Modélisation</h3>
                <h1 style="color:#2e7b32;">{prediction:,.0f} DH</h1>
                <p>Modèle utilisé : Arbre de Décision (Régression sur Log)</p>
            </div>
        """, unsafe_allow_html=True)