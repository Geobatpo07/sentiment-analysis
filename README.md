# Projet d'Analyse de Sentiment

Ce projet implémente une application de classification de sentiment utilisant des algorithmes d'apprentissage automatique (Perceptron, Average Perceptron, Pegasos) avec une représentation *bag-of-words*. L'application permet d'analyser le sentiment d'avis (positif ou négatif) et fournit une interface web conviviale grâce à Flask.

## Table des Matières

- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
    - [Exécution en Console](#exécution-en-console)
    - [Interface Web avec Flask](#interface-web-avec-flask)
- [Tests](#tests)
- [Contribuer](#contribuer)
- [Licence](#licence)

## Fonctionnalités

- **Chargement et prétraitement des données :**  
  Chargement des fichiers TSV contenant les avis et extraction des caractéristiques via une représentation *bag-of-words* (avec ou sans binarisation).

- **Implémentation de plusieurs algorithmes :**  
  Perceptron, Average Perceptron et Pegasos pour la classification de sentiment.

- **Tuning d'hyperparamètres :**  
  Évaluation et sélection des meilleurs hyperparamètres via des courbes de précision sur les ensembles d'entraînement et de validation.

- **Interprétabilité du modèle :**  
  Extraction des mots les plus explicatifs (ceux ayant les plus forts poids dans le modèle).

- **Interface Web :**  
  Interface web conviviale développée avec Flask.

## Prérequis

- Python 3.6 ou supérieur
- Les packages listés dans `requirements.txt` (ex. : `numpy`, `matplotlib`, `Flask`, `FastAPI`, `uvicorn`, etc.)

## Installation

1. **Cloner le dépôt :**
  ```bash
   git clone https://github.com/Geobatpo07/sentiment-analysis.git
   cd sentiment-analysis
 ```
   
2. **Créer un environnement virtuel :**
  ```bash
    python3 -m venv env
    source env/bin/activate  # Sur Windows: env\Scripts\activate
  ```

3. **Installer les dépendances :**
  ```bash
    pip install -r requirements.txt
  ```

## Utilisation
#### Exécution en Console:
```bash
  python main.py
```

#### Interface Web avec Flask
```bash
  python app/flask_app.py
```

## Tests
```bash
  pytest tests/
```

## Contribuer

Les contributions sont les bienvenues ! 

Pour contribuer :

1. Forkez le dépôt.
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/ma-fonctionnalite`).
3. Commitez vos changements (`git commit -m "Description de ma fonctionnalité"`).
4. Poussez la branche (`git push origin feature/ma-fonctionnalite`).
5. Ouvrez une Pull Request.

## Licence

Ce projet est distribué sous licence MIT. Consultez le fichier `LICENSE` pour plus d'informations.