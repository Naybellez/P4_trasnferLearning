"""
Empirical random-chance simulation for the "predict north angle" task.

Compares:
1. Theoretical chance level (assumes labels uniform across 0-359)
2. Empirical chance level using YOUR actual label distribution
   (uniform random argmax as the "prediction")
3. Empirical chance level using fully random 360-length noise vectors
   (closer to how a real model's output behaves)

Handles circular wraparound near 0/360 correctly.
"""

import numpy as np



def circular_distance(a, b, n):
    """Shortest distance between two positions on a circle of size n."""
    d = np.abs(a - b)
    return np.minimum(d, n - d)


def is_hit(pred_pos, true_pos, tolerance, n):
    return circular_distance(pred_pos, true_pos, n) <= tolerance


def simulate_uniform_argmax(true_labels, n_sims, tolerance, n):
    """
    true_labels: array of true label positions (one per image/trial),
                 e.g. shape (n_images,) with integer values 0-359.
    Draws a uniformly random position as the 'prediction' each time.
    """
    n_images = len(true_labels)
    accuracies = []
    for _ in range(n_sims):
        random_preds = np.random.randint(0, n, size=n_images)
        hits = is_hit(random_preds, np.array(true_labels), tolerance, n)
        accuracies.append(hits.mean())
    return np.array(accuracies)


def simulate_random_vector_argmax(true_labels, n_sims, tolerance, n):
    """
    More faithful to a real model: generate random noise as the full
    360-length prediction vector, take its argmax as the prediction.
    For pure random noise this should converge to the same result as
    simulate_uniform_argmax, but it's a good sanity check /
    closer match to your actual pipeline if you want to feed in
    e.g. an untrained network's real outputs instead of noise.
    """
    n_images = len(true_labels)
    accuracies = []
    for _ in range(n_sims):
        random_vectors = np.random.rand(n_images, n)
        random_preds = np.argmax(random_vectors, axis=1)
        hits = is_hit(random_preds, np.array(true_labels), tolerance, n)
        accuracies.append(hits.mean())
    return np.array(accuracies)

