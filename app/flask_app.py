# app/flask_app.py
from flask import Flask, render_template
from pathlib import Path
import src.project as p1
import src.utils as utils
import numpy as np

app = Flask(__name__)

#---------------------------------------------------------------------------
# Construction des chemins d'accès aux fichiers de données avec pathlib
#---------------------------------------------------------------------------
# Le répertoire courant de ce fichier (app/)
current_dir = Path(__file__).resolve().parent
# Le répertoire 'data' se trouve à la racine du projet (niveau supérieur à app/)
data_dir = current_dir.parent / 'data'

# Chargement des données à partir des fichiers TSV
train_data = utils.load_data(str(data_dir / 'reviews_train.tsv'))
val_data   = utils.load_data(str(data_dir / 'reviews_val.tsv'))
test_data  = utils.load_data(str(data_dir / 'reviews_test.tsv'))

# Extraction des textes et labels
train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

# Création du dictionnaire avec retrait des stopwords
dictionary = p1.bag_of_words(train_texts, remove_stopword=True)

# Extraction des features (avec comptage, binarize=False)
train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary, binarize=False)
val_bow_features   = p1.extract_bow_feature_vectors(val_texts, dictionary, binarize=False)
test_bow_features  = p1.extract_bow_feature_vectors(test_texts, dictionary, binarize=False)

#---------------------------------------------------------------------------
# Entraînement du modèle et évaluation sur le jeu de test
#---------------------------------------------------------------------------
optimal_T = 25
optimal_L = 0.01

theta, theta_0 = p1.pegasos(train_bow_features, train_labels, optimal_T, optimal_L)
test_predictions = p1.classify(test_bow_features, theta, theta_0)
test_accuracy = p1.accuracy(test_predictions, test_labels)

# Extraction des mots les plus explicatifs
best_theta = theta.copy()  # Les poids appris (sans le biais)
# Construction de la wordlist à partir du dictionnaire
wordlist = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)

#---------------------------------------------------------------------------
# Route principale de l'interface web (dashboard)
#---------------------------------------------------------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html",
                           test_accuracy=test_accuracy,
                           optimal_T=optimal_T,
                           optimal_L=optimal_L,
                           explanatory_words=sorted_word_features[:10])

if __name__ == "__main__":
    app.run(debug=True)
