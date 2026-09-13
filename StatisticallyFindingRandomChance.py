
import numpy as np
from StatisticallyFindingRandomChanceFUNCs import simulate_uniform_argmax, simulate_random_vector_argmax, is_hit
import pandas as pd

all_df = pd.read_pickle("/media/noli/Expansion/P4_saves/R1/results_pkls/Selected_cols/all_df.pkl")
just_labels = all_df['test_labels']
true_labels = [np.array(i).argmax() for i in just_labels]

N_POSITIONS = 360
TOLERANCE = 23          # positions either side of true label
N_SIMULATIONS = len(true_labels)   # number of random draws to average over



# --- Example 1: labels uniformly spread across all 360 positions ---
np.random.seed(0)
uniform_labels = np.random.randint(0, N_POSITIONS, size=2000)

acc_uniform = simulate_uniform_argmax(true_labels, n_sims=N_SIMULATIONS, tolerance=TOLERANCE, n=N_POSITIONS)
print("=== Labels uniformly distributed across 0-359 ===")
print(f"Theoretical chance (47/360):    {47/360:.4f}")
print(f"Empirical chance (mean):        {acc_uniform.mean():.4f}")
print(f"Empirical chance (95% CI):      "
        f"[{np.percentile(acc_uniform, 2.5):.4f}, "
        f"{np.percentile(acc_uniform, 97.5):.4f}]")
print()

# --- Example 2: labels clustered (e.g. only ~10 distinct true angles,
#     repeated across many images from the same transect location) ---
ten_fixed_angles = np.random.choice(N_POSITIONS, size=10, replace=False)
clustered_labels = np.random.choice(ten_fixed_angles, size=2000)

acc_clustered = simulate_uniform_argmax(clustered_labels, n_sims=N_SIMULATIONS, tolerance=TOLERANCE, n=N_POSITIONS)
print("=== Labels clustered around 10 fixed transect angles ===")
print(f"Theoretical chance (47/360):    {47/360:.4f}")
print(f"Empirical chance (mean):        {acc_clustered.mean():.4f}")
print(f"Empirical chance (95% CI):      "
        f"[{np.percentile(acc_clustered, 2.5):.4f}, "
        f"{np.percentile(acc_clustered, 97.5):.4f}]")
print()
print("(This block shows how clustering CAN shift empirical chance,")
print(" though pure uniform-random guessing vs clustered labels should")
print(" still land close to 47/360 on average - true skew usually comes")
print(" from a MODEL that isn't guessing uniformly, e.g. one that collapses")
print(" toward a constant/narrow output that happens to overlap label clusters.")
print(" See 'degenerate baseline' test below for that scenario.)")
print()

# --- Example 3: "degenerate" baseline - model always predicts the
#     same fixed angle (or a narrow range) regardless of input.
#     This simulates a model that has NOT learned anything useful
#     but happens to exploit clustered/non-uniform labels. ---
degenerate_pred = np.full(2000, fill_value=ten_fixed_angles[0])
hits_degenerate = is_hit(degenerate_pred, clustered_labels, TOLERANCE, N_POSITIONS)
print("=== Degenerate baseline: always predicts one fixed angle ===")
print(f"Accuracy against clustered labels: {hits_degenerate.mean():.4f}")
print("If this is well above your theoretical chance level, it strongly")
print("suggests your 'poor' models are exploiting non-uniform label")
print("structure rather than actually learning anything.")

# if a model always chooses the same angle, and our data is not evenly distributed (which it is not)
# RANDOM CHANCE LEVEL  is 0.3150 or 31.5%  could round down to 30%, 31% or round up to 32%