# OR Pointage Analyzer

Système intelligent de contrôle des pointages et d’analyse d’efficience des Ordres de Réparation (OR).

Ce projet permet de détecter automatiquement :

- 🔎 Anomalies techniques de pointage  
- ⏳ Anomalies de process administratif (retards, absence de pointage)  
- 💰 Anomalies d’efficience financière  
- 🤖 Patterns cachés via Machine Learning  

Le système est conçu pour être :
- Modulaire
- Réutilisable multi-années (2024, 2025, etc.)
- Déployable sur Streamlit Cloud
- Prêt pour industrialisation future

---

## 🎯 Objectif

Transformer l’analyse des pointages d’un simple contrôle manuel vers un système intelligent de pilotage performance service.

Ce projet combine :

- Données opérationnelles (IE)
- Données de pointage détaillées
- Données financières (BO)
- Règles métier
- Machine Learning (Isolation Forest)

---

## 📊 Sources de données

L’application nécessite 3 fichiers CSV :

### 1️⃣ IE (Extraction OR)
Contient :
- OR
- Date création
- Type OR
- Client
- Localisation
- Matériel
- Intervention

Inclut OR ouverts et clôturés.

---

### 2️⃣ Pointage détaillé
Contient :
- OR
- Technicien
- Date pointage
- Heures
- Type activité

---

### 3️⃣ BO (Business Objects)
Contient uniquement OR clôturés et facturés :

- Temps vendu (OR)
- Temps prévu devis (OR)
- Durée pointage total
- Montant MO
- Montant total
- Date clôture
- Date facture
- Date dernier pointage

---

## 🧠 Architecture du projet
or-performance-analyzer/

│

├── app.py
├── scoring/
│ ├── preprocess.py
│ ├── feature_engineering.py
│ ├── rule_engine.py
│ ├── ml_model.py
│ └── scorer.py
│
├── models/
├── requirements.txt
└── README.md


---

## 🏗️ Fonctionnement global

1. Upload des 3 datasets
2. Nettoyage et prétraitement
3. Feature engineering avancé
4. Détection d’anomalies rule-based
5. Détection d’anomalies ML (Isolation Forest)
6. Calcul des scores :
   - anomaly_score_technique
   - anomaly_score_process
   - anomaly_score_financier
   - anomaly_score_global
7. Export des résultats enrichis

---

## 🔎 Dimensions d’analyse

### 1️⃣ Technique
- Heures excessives
- Variance anormale
- Z-score élevé

### 2️⃣ Process
- OR sans pointage
- Retard après deadline administrative
- Délai anormal premier pointage

### 3️⃣ Financière
- Surconsommation
- Efficience < seuil
- Marge négative

---

## 🤖 Machine Learning

Le système utilise :

- Isolation Forest
- Normalisation des variables
- Score d’anomalie entre 0 et 100
- Modèle sauvegardé avec joblib

Le modèle est :
- Réutilisable
- Robuste aux nouvelles catégories
- Adaptable aux nouvelles années

---

## 🚀 Déploiement

### Déploiement Streamlit Cloud

1. Push sur GitHub
2. Connecter le repo à Streamlit Cloud
3. Sélectionner `app.py`
4. Deploy

L’application devient accessible via URL publique.

---

## 🛡️ Sécurité des données

Version actuelle :

- Aucun stockage permanent
- Les données sont traitées en mémoire
- Aucun historique conservé


---

## 🔧 Installation locale

```bash
pip install -r requirements.txt
streamlit run app.py
📈 Évolution future

Ajout Autoencoder

Base SQLite pour historique

Dashboard Power BI connecté

API FastAPI

Containerisation Docker

Intégration ERP

🏆 Vision

Passer d’un contrôle manuel des pointages à un système intelligent de pilotage performance service.

Ce projet peut évoluer vers :

Outil interne corporate

Standard groupe

Solution industrialisable

👤 Auteur
Mohamadou Moustapha GAYE
Adjoint Méthodes & Process
Spécialisation : optimisation performance service & data intelligence



