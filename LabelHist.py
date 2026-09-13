import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


all_df=pd.read_pickle("/media/noli/Expansion/P4_saves/R1/results_pkls/Selected_cols/all_df2.pkl")

justLabels = all_df['test_labels']
print(justLabels[0])

"""print(type(justLabels))
print(justLabels)
print(justLabels.shape)s
print(justLabels[0])"""

max_labs = [np.array(i).argmax() for i in justLabels]
print(max_labs[0])

plt.hist(max_labs, bins=range(0, 360), edgecolor='black')
plt.savefig("/media/noli/Expansion/P4_saves/R1/results_pkls/Selected_cols/TestLabelDistHist.png", dpi=500)
plt.show()