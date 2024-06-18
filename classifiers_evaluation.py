from matplotlib import pyplot as plt

from classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import numpy as np


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for name, filename in [("Linearly Separable", "linearly_separable.npy"),
                           ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(filename)
        losses = []

        def save_loss_callback(perceptron, x_i, y_i):
            loss = perceptron.loss(X, y)
            losses.append(loss)

        perceptron = Perceptron(callback=save_loss_callback).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(data=go.Scatter(x=list(range(len(losses))), y=losses, mode="lines", marker=dict(color="black")),
                        layout=go.Layout(
                            title={"text": f"Perceptron Training Error for the {name} Dataset", "x": 0.5},
                            xaxis={"title": "Iteration", "range": [0, len(losses)]},  # Adjust x-axis range
                            yaxis={"title": "Misclassification Error"}))

        fig.show()
        # Added in order to see the plotted graphs on my personal computer
        fig.write_html(f"perceptron_fit_{name}.html")

        # fig.write_image(f"perceptron_fit_{name}.png")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """

    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for filename in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(filename)

        # Fit models and predict over training set
        lda_model = LDA()
        naive_bayes_model = GaussianNaiveBayes()

        lda_model.fit(X, y)
        naive_bayes_model.fit(X, y)

        lda_predictions = lda_model.predict(X)
        naive_bayes_predictions = naive_bayes_model.predict(X)

        # Calculate accuracies
        from loss_functions import accuracy
        lda_accuracy = accuracy(y, lda_predictions) * 100
        naive_bayes_accuracy = accuracy(y, naive_bayes_predictions) * 100

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            f"Gaussian Naive Bayes (Accuracy: {naive_bayes_accuracy:.2f}%)",
            f"LDA (Accuracy: {lda_accuracy:.2f}%)"
        ], horizontal_spacing=0.1)

        # Add traces for data-points setting symbols and colors according to class
        # For Naive Bayes
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1],
                                 mode="markers",
                                 marker=dict(color=naive_bayes_predictions, symbol=y,
                                             colorscale=["blue", "red"]),
                                 name="Naive Bayes Predictions", showlegend=False),
                      row=1, col=1)

        # For LDA
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1],
                                 mode="markers",
                                 marker=dict(color=lda_predictions, symbol=y,
                                             colorscale=["blue", "red"]),
                                 name="LDA Predictions", showlegend=False),
                      row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        # For Naive Bayes
        fig.add_trace(go.Scatter(x=naive_bayes_model.mu_[:, 0], y=naive_bayes_model.mu_[:, 1],
                                 mode="markers", marker=dict(color="black", symbol="x", size=10),
                                 name="Means Naive Bayes", showlegend=False),
                      row=1, col=1)

        # For LDA
        fig.add_trace(go.Scatter(x=lda_model.mu_[:, 0], y=lda_model.mu_[:, 1],
                                 mode="markers", marker=dict(color="black", symbol="x", size=10), name="Means LDA",
                                 showlegend=False),
                      row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        # For Naive Bayes
        for mean, cov in zip(naive_bayes_model.mu_, naive_bayes_model.vars_):
            ellipse = get_ellipse(mean, np.diag(cov))
            ellipse.update(showlegend=False)  # Set showlegend=False for the ellipse
            fig.add_trace(ellipse, row=1, col=1)

        # For LDA
        for mean, cov in zip(lda_model.mu_, [lda_model.cov_] * len(lda_model.classes_)):
            ellipse = get_ellipse(mean, cov)
            ellipse.update(showlegend=False)

            fig.add_trace(ellipse, row=1, col=2)

        fig.update_layout(title_text=f"Comparison of Gaussian Classifiers on {filename}")
        fig.show()
        # Added in order to see the plotted graphs on my personal computer
        fig.write_html(f"gaussian_comparison_{filename}.html")
        # fig.write_image(f"gaussian_comparison_{filename}.png")



if __name__ == '__main__':
    np.random.seed(500)
    run_perceptron()
    compare_gaussian_classifiers()
