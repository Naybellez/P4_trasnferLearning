import pandas as pd
import os


mainPath = "/media/noli/Expansion/P4_saves/R1/"
pkl_folder = "results_pkls/"

for file in os.listdir(mainPath+pkl_folder):
    if file.endswith(".pkl"):
        df = pd.read_pickle(mainPath+pkl_folder+file)
        print(file, df.shape)
        print(df.head())
        break