
import pandas as pd
import os

def oneBIGdf(pickle_path:str, modelnames:list):
    hold = []
    for modelname in modelnames:
        file_name = f"{modelname}_SC2.pkl"
        file_path = os.path.join(pickle_path, file_name)
        
        if os.path.exists(file_path):
            print(f"Loading {file_name}...")
            df = pd.read_pickle(file_path)
            print(f"Dataframe for {file_name} loaded. Shape: {df.shape}")
            hold.append(df)

    merged = pd.concat(hold, ignore_index=True, sort=False)
    print(f"merged shape: {merged.shape}")
    return merged

