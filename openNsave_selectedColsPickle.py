import pandas as pd
from OpenSelectedPicklesFuncs import oneBIGdf

pickle_path = "/media/noli/Expansion/P4_saves/R1/results_pkls/Selected_cols/"

all_df = oneBIGdf(pickle_path, ['2c2l','3c2l','4c3l','6c3l','7c3l','8c3l'])

all_df.to_pickle(f"{pickle_path}all_df2.pkl")

