# JunbiData – Ready your dataset

**Junbi** signifie "préparation" en japonais.  
Cette application a pour but de **visualiser, nettoyer et transformer** un jeu de données pour le rendre prêt à l'entraînement en machine learning.

---

## 🚀 Fonctionnalités principales

- 📊 Visualisation du taux de complétion (lignes / colonnes)
- 📈 Distribution des valeurs (histogrammes, boxplots, barplots)
- 🧹 Nettoyage des doublons et outliers (détection + action)
- 🔍 Détection automatique des types de colonnes + conversion manuelle
- 🔠 Encodage des colonnes catégorielles (One-Hot / Ordinal)
- ⚖️ Application de scalers (Standard, MinMax, Robust, Normalizer)
- 🧪 Réduction de dimension via PCA

---

## 📁 Structure du projet (provisoire)
junbidata/
│
├── app/
│   ├── main.py                 # Interface principale, onglets, layout
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── chargement.py       # Module de chargement (upload)
│   │   └── visualisation.py    # Module de visualisation (taux de complétion, outliers, ...)
├── assets/                     # Fichiers CSS personnalisés
│   └── custom.css
├── README.md
├── requirements.txt
└── run.py

## Librairies utilisées
Dash
Plotly
Pandas

### Pour les datasets > 30 Mo : Spark
Utilisation de WSL ou Linux conseillé (pour windows hadoop 3.3.6 max pour winutils.exe dispo ici : https://github.com/cdarlint/winutils)
Spark
Pyspark
Hadoop