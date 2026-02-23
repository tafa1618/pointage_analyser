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
