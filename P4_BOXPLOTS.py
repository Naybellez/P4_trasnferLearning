from Plot_P4 import plot_boxplots, plot_boxplot2
from openNsave_selectedColsPickle import all_df
import matplotlib.pyplot as plt

# chance = 46 / 360 ≈ 0.1278, or about 12.78% rounding up to 13%
# if model always chooses the same direction and our data is not even (which it is not - see true label histogram)
# RANDOM CHANCE LEVEL IS: 0.3150 or 31.5%  rounded up to 32%

# theoretical chance = 47/360 = 0.1306 = 13%
 
AlwaysGuessC_level = 32
theoretical_chance = 13

save_loc = "/media/noli/Expansion/P4_saves/R1/results_pkls/Selected_cols/"
#plot_boxplots(all_df, 'TESTAccBase', randomChanceLevel=random_chance_level)
#plot_boxplots(all_df, 'TESTAccPeakDist', randomChanceLevel=random_chance_level)
#print(all_df['model_name'].unique())
plot_boxplot2(all_df,'TESTAccBase', ChanceL1=AlwaysGuessC_level, ChanceL2=theoretical_chance)
plt.savefig(save_loc+"P4_BASEACC_box.png", dpi=500)
plt.show()
plot_boxplot2(all_df,'TESTAccPeakDist')
plt.savefig(save_loc+"P4_PEAKDIST_box.png",dpi=500)
plt.show()