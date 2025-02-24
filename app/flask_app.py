# app/flask_app.py
from flask import Flask, render_template, request, redirect, url_for
from pathlib import Path
import src.project as p1
import src.utils as utils
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from functools import lru_cache
import gc
import io
import base64
from wordcloud import WordCloud

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Construct paths using pathlib
# ---------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / 'data'

# ---------------------------------------------------------------------------
# Cache the data loading to avoid reloading on every request
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_all_data():
    train_data = utils.load_data(str(data_dir / 'reviews_train.tsv'))
    val_data = utils.load_data(str(data_dir / 'reviews_val.tsv'))
    test_data = utils.load_data(str(data_dir / 'reviews_test.tsv'))

    train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
    val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
    test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_all_data()

# Build dictionary (with stopwords removed)
dictionary = p1.bag_of_words(train_texts, remove_stopword=True)

# Extract bag-of-words features (using counts; binarize=False) and convert to float32
train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary).astype(np.float32)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary).astype(np.float32)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary).astype(np.float32)

# Train a model using Pegasos with fixed optimal hyperparameters (for demonstration)
optimal_T = 25
optimal_L = 0.01
theta, theta_0 = p1.pegasos(train_bow_features, train_labels, optimal_T, optimal_L)
test_predictions = p1.classify(test_bow_features, theta, theta_0)
test_accuracy = round(p1.accuracy(test_predictions, test_labels), 2)

# Extract most explanatory words (using the learned weights)
best_theta = theta.copy()  # We use weights only (without bias)
# Build wordlist by sorting dictionary items by index
wordlist = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)

# ---------------------------------------------------------------------------
# Helper: Generate interactive Plotly plot as HTML
# ---------------------------------------------------------------------------
def generate_plotly_plot(algo_name, param_name, param_vals, train_acc, val_acc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=param_vals, y=train_acc, mode="markers+lines", name="Train Accuracy"))
    fig.add_trace(go.Scatter(x=param_vals, y=val_acc, mode="markers+lines", name="Validation Accuracy"))
    fig.update_layout(
        title=f"{algo_name} Tuning ({param_name})",
        xaxis_title=param_name,
        yaxis_title="Accuracy (%)",
        template="plotly_white"
    )
    return pio.to_html(fig, full_html=False)

# ---------------------------------------------------------------------------
# Helper: Generate interactive Plotly plot as HTML for toy data
# ---------------------------------------------------------------------------
def generate_toy_plot(algo="Pegasos"):
    toy_data_path = data_dir / 'toy_data.tsv'
    toy_features, toy_labels = utils.load_toy_data(str(toy_data_path))

    if algo == "Pegasos":
        T_toy = 10
        L_toy = 0.2
        thetas = p1.pegasos(toy_features, toy_labels, T_toy, L_toy)
    elif algo == "Perceptron":
        T_toy = 10
        thetas = p1.perceptron(toy_features, toy_labels, T_toy)
    elif algo == "Average Perceptron":
        T_toy = 10
        thetas = p1.average_perceptron(toy_features, toy_labels, T_toy)
    else:
        T_toy = 10
        L_toy = 0.2
        thetas = p1.pegasos(toy_features, toy_labels, T_toy, L_toy)

    fig = go.Figure()
    colors = ['blue' if label == 1 else 'red' for label in toy_labels]
    fig.add_trace(go.Scatter(
        x=toy_features[:,0],
        y=toy_features[:,1],
        mode='markers',
        marker=dict(color=colors),
        name='Data Points'
    ))

    x_min = float(toy_features[:,0].min())
    x_max = float(toy_features[:,0].max())
    x_vals = np.linspace(x_min, x_max, 100)
    if abs(thetas[0][1]) > 1e-6:
        y_vals = -(thetas[0][0] * x_vals + thetas[1]) / thetas[0][1]
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='Decision Boundary'
        ))
    else:
        fig.add_annotation(text="Decision boundary undefined", showarrow=False)

    fig.update_layout(
        title=f"Toy Data Classification ({algo})",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        template="plotly_white"
    )
    return pio.to_html(fig, full_html=False)

# ---------------------------------------------------------------------------
# Helper: Generate a word cloud image (as base64) from the dataset
# ---------------------------------------------------------------------------
def generate_wordcloud():
    # Compute frequency from the bag-of-words features
    freq_vector = np.sum(train_bow_features, axis=0)  # Sum over all documents
    freq_vector = np.asarray(freq_vector).flatten()
    # Build frequency dictionary using the wordlist (assumed to be in the same order as dictionary columns)
    freq_dict = {word: int(freq_vector[i]) for i, word in enumerate(wordlist)}

    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(freq_dict)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    return img_base64

wordcloud_img = generate_wordcloud()

# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------
@app.route("/")
def dashboard():
    # Get the toy algorithm selection from query parameters (default: Pegasos)
    toy_algo = request.args.get("toy_algo", "Pegasos")
    toy_plot_html = generate_toy_plot(toy_algo)
    return render_template("dashboard.html",
                           test_accuracy=test_accuracy,
                           optimal_T=optimal_T,
                           optimal_L=optimal_L,
                           explanatory_words=sorted_word_features[:10],
                           toy_plot=toy_plot_html,
                           toy_algo=toy_algo,
                           wordcloud_img=wordcloud_img)

@app.route("/run_algorithm", methods=["GET", "POST"])
def run_algorithm():
    if request.method == "POST":
        selected_algo = request.form.get("algorithm")
        if selected_algo:
            tuning_results = run_selected_algorithm(selected_algo)
            gc.collect()
            return render_template("tuning.html", tuning_results=tuning_results, selected_algo=selected_algo)
        else:
            return "No algorithm selected.", 400
    return render_template("run_algorithm.html")

def run_selected_algorithm(algo):
    data = (train_bow_features, train_labels, val_bow_features, val_labels)
    Ts = [1, 5, 10, 15, 25, 50]
    Ls = [0.001, 0.01, 0.1, 1, 10]
    result = {}
    if algo == "Perceptron":
        tune_results = utils.tune_perceptron(Ts, *data)
        best_param = Ts[np.argmax(tune_results[1])]
        plot_html = generate_plotly_plot("Perceptron", "T", Ts, tune_results[0], tune_results[1])
        result = {"Perceptron": {"best_param": best_param, "param": "T", "plot": plot_html}}
    elif algo == "Average Perceptron":
        tune_results = utils.tune_avg_perceptron(Ts, *data)
        best_param = Ts[np.argmax(tune_results[1])]
        plot_html = generate_plotly_plot("Average Perceptron", "T", Ts, tune_results[0], tune_results[1])
        result = {"Average Perceptron": {"best_param": best_param, "param": "T", "plot": plot_html}}
    elif algo == "Pegasos":
        fix_L = 0.01
        tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
        best_T = Ts[np.argmax(tune_results_T[1])]
        tune_results_L = utils.tune_pegasos_L(best_T, Ls, *data)
        best_L = Ls[np.argmax(tune_results_L[1])]
        plot_html_T = generate_plotly_plot("Pegasos", "T", Ts, tune_results_T[0], tune_results_T[1])
        plot_html_L = generate_plotly_plot("Pegasos", "L", Ls, tune_results_L[0], tune_results_L[1])
        result = {
            "Pegasos (Tuning T)": {"best_param": best_T, "param": "T", "plot": plot_html_T},
            "Pegasos (Tuning L)": {"best_param": best_L, "param": "L", "plot": plot_html_L}
        }
    return result

@lru_cache(maxsize=1)
def get_tuning_results():
    data = (train_bow_features, train_labels, val_bow_features, val_labels)
    Ts = [1, 5, 10, 15, 25, 50]
    Ls = [0.001, 0.01, 0.1, 1, 10]
    pct_tune_results = utils.tune_perceptron(Ts, *data)
    avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
    fix_L = 0.01
    peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
    fix_T = Ts[np.argmax(peg_tune_results_T[1])]
    peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)

    def generate_plotly(algo_name, param_name, param_vals, train_acc, val_acc):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=param_vals, y=train_acc, mode="markers+lines", name="Train Accuracy"))
        fig.add_trace(go.Scatter(x=param_vals, y=val_acc, mode="markers+lines", name="Validation Accuracy"))
        fig.update_layout(
            title=f"{algo_name} Tuning ({param_name})",
            xaxis_title=param_name,
            yaxis_title="Accuracy (%)",
            template="plotly_white"
        )
        return pio.to_html(fig, full_html=False)

    perc_plot = generate_plotly("Perceptron", "T", Ts, pct_tune_results[0], pct_tune_results[1])
    avg_perc_plot = generate_plotly("Average Perceptron", "T", Ts, avg_pct_tune_results[0], avg_pct_tune_results[1])
    peg_tune_T_plot = generate_plotly("Pegasos", "T", Ts, peg_tune_results_T[0], peg_tune_results_T[1])
    peg_tune_L_plot = generate_plotly("Pegasos", "L", Ls, peg_tune_results_L[0], peg_tune_results_L[1])

    best_perc_T = Ts[np.argmax(pct_tune_results[1])]
    best_avg_perc_T = Ts[np.argmax(avg_pct_tune_results[1])]
    best_peg_T = Ts[np.argmax(peg_tune_results_T[1])]
    best_peg_L = Ls[np.argmax(peg_tune_results_L[1])]

    return {
        "Perceptron": {"best_param": best_perc_T, "param": "T", "plot": perc_plot},
        "Average Perceptron": {"best_param": best_avg_perc_T, "param": "T", "plot": avg_perc_plot},
        "Pegasos (Tuning T)": {"best_param": best_peg_T, "param": "T", "plot": peg_tune_T_plot},
        "Pegasos (Tuning L)": {"best_param": best_peg_L, "param": "L", "plot": peg_tune_L_plot}
    }

gc.collect()

if __name__ == "__main__":
    app.run(debug=True)
