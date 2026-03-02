# 📱 Projet d’Analyse des Données Smartphones

Application de data science pour explorer, nettoyer et modéliser les prix de smartphones, avec une interface interactive Streamlit pour tester des prédictions en direct.

---

## 🎯 Objectifs du projet

- Créer un dataset de smartphones (avec quelques anomalies volontairement ajoutées)
- Effectuer l’exploration et le prétraitement des données
- Visualiser les données (distribution, corrélation, PCA, t-SNE)
- Entraîner un modèle de régression (Arbre de Décision)
- Prédire le prix d’un smartphone à partir de ses caractéristiques

---

## 🧱 Structure du projet

```text
Projet_Analyse_Donnees_Smartphones/
├── data/
│   └── smartphones.csv
└── src/
    ├── app.py
    ├── modeling.py
    ├── preprocessing.py
    └── scraping.py
```

---

## ⚙️ Prérequis

- Python 3.9+ (recommandé)
- pip

### Dépendances Python

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

---

## 🚀 Lancer le projet

### 1) Générer / régénérer le dataset
```bash
python src/scraping.py
```

### 2) Exécuter l’analyse et la visualisation hors interface
```bash
python src/preprocessing.py
python src/modeling.py
```

### 3) Lancer l’application Streamlit
```bash
streamlit run src/app.py
```

---

## 🧠 Fonctionnalités de l’application

Dans la barre latérale Streamlit, vous pouvez naviguer entre :

1. **Exploration & Nettoyage**
   - Distribution des prix
   - Heatmap des corrélations

2. **Réduction Dimensionnelle**
   - Projection PCA (2D)
   - Visualisation t-SNE (2D)

3. **Analyse de l’Arbre**
   - Importance des variables
   - Visualisation de l’arbre de décision

4. **Prédiction en Direct**
   - Saisie des caractéristiques d’un smartphone
   - Estimation automatique du prix en DH

---

## 📊 Variables utilisées

- **Brand** : marque
- **Model** : modèle
- **Price_DH** : prix en dirhams
- **RAM_GB** : mémoire RAM
- **Storage_GB** : stockage
- **Screen_Size_Inch** : taille écran
- **Battery_mAh** : capacité batterie

---

## 🧹 Nettoyage des données

Le pipeline inclut notamment :

- Imputation des valeurs manquantes (médiane / moyenne)
- Suppression d’anomalies extrêmes de prix
- Encodage one-hot de la marque
- Standardisation des variables numériques
- Transformation logarithmique de la cible pour stabiliser la régression

---

## 🔍 Modèle ML

- **Algorithme** : `DecisionTreeRegressor`
- **Objectif** : prédire le prix smartphone
- **Sortie finale** : conversion inverse du log pour afficher un prix réel en DH

---

## 🛠️ Améliorations possibles

- Évaluation plus complète (MAE, RMSE, validation croisée)
- Comparaison avec d’autres modèles (Random Forest, XGBoost, etc.)
- Sauvegarde/versioning du modèle
- Déploiement web (Streamlit Cloud, Docker)

---

## 👤 Auteur

Projet réalisé dans le cadre d’un exercice d’analyse de données et de modélisation ML.
