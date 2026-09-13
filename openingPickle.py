import pandas as pd
import os

pickle_path = "/media/noli/Expansion/P4_saves/R1/results_pkls/All_data/"
save_path = "/media/noli/Expansion/P4_saves/R1/results_pkls/Selected_cols/"

def load_pickles(pickle_path, modelname, resolution):
    file_name = f"{modelname}_{resolution}.pkl"
    file_path = os.path.join(pickle_path, file_name)
    
    if os.path.exists(file_path):
        print(f"Loading {file_name}...")
        df = pd.read_pickle(file_path)
        print(f"Dataframe for {file_name} loaded. Shape: {df.shape}")
        return df
    else:
        print(f"File {file_name} does not exist in the specified path.")
        return None

def load_all_pickles(pickle_path, save_path, modelname):
    resolutions = ['452', '226', '113', '57', '29']
    dataframes = []
    
    for res in resolutions:
        df = load_pickles(pickle_path, modelname, res)
        if df is not None:
            dataframes.append(df)
    
    if dataframes:
        all_df = pd.concat(dataframes, ignore_index=True)
        print(f"All dataframes for {modelname} concatenated. Shape: {all_df.shape}")
        print("Selecting specific columns...")
        all_df_selectedCols = all_df[['model_name', 'resolution','seed', 't_labels','t_loss_list', 'v_loss_list', 'TESTAccBase','TESTAccPeakDist','test_predict', 'test_labels']]
        print(list(all_df_selectedCols))
        print(f"Saving selected columns for {modelname}...")
        all_df_selectedCols.to_pickle(f"{save_path}{modelname}_SC2.pkl")
        return None
    else:
        print(f"No dataframes found for {modelname}.")
        return None

mods = ["2c2l", "3c2l", "4c3l", '6c3l', '7c3l', '8c3l'] # 
for mod in mods:
    load_all_pickles(pickle_path, save_path, mod)
