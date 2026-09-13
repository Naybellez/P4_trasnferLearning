from Jsons2Pickle_funcs import  sort_jsons,readin_json_merge_all
import pandas as pd
import pickle


mainPath = "/media/noli/Expansion/P4_saves/R1/"

folder_2c = "2c2l/R1_2c2l_transfer_300E/"
folder_3c = "3c2l/R1_3c2l_transfer_300E/"
folder_4c = "4c3l/R1_4c3l_transfer_300E/"
folder_6c = "6c3l/R1_6c3l_transfer_300E/"
folder_7c = "7c3l/R1_7c3l_transfer_300E/"
folder_8c = "8c3l/R1_8c3l_transfer_300E/"

save_folder = "results_pkls/"

folders = [folder_2c, folder_3c, folder_4c, folder_6c, folder_7c, folder_8c]
resolutions = ['[452, 144]', '[226, 72]', '[113, 36]', '[57, 18]', '[29, 9]']

def get_DFs(mainPath, folder, resolutions, modelname):
    sorted_452, sorted_226, sorted_113, sorted_57, sorted_29 = sort_jsons(mainPath, folder, resolutions)

    df_452 = readin_json_merge_all(mainPath+folder, sorted_452)
    print(f"dataframe for {modelname} 452 created. about to pickle...")
    df_452.to_pickle(f"{mainPath+save_folder}{modelname}_452.pkl")
    df_226 = readin_json_merge_all(mainPath+folder, sorted_226)
    print(f"dataframe for {modelname}  226 created. about to pickle...")
    df_226.to_pickle(f"{mainPath+save_folder}{modelname}_226.pkl")
    df_113 = readin_json_merge_all(mainPath+folder, sorted_113)
    print(f"dataframe for {modelname} 113  created. about to pickle...")
    df_113.to_pickle(f"{mainPath+save_folder}{modelname}_113.pkl")
    """df_57 = readin_json_merge_all(mainPath+folder, sorted_57)
    print(f"dataframe for {modelname} 57 created. about to pickle...")
    df_57.to_pickle(f"{mainPath+save_folder}{modelname}_57.pkl")
    df_29 = readin_json_merge_all(mainPath+folder, sorted_29)
    print(f"dataframe for {modelname} 29 created. about to pickle...")
    df_29.to_pickle(f"{mainPath+save_folder}{modelname}_29.pkl")"""

    print("Done")

"""#df_452, df_226, df_113, df_57, df_29 = get_DFs(mainPath, folder_2c, resolutions,index=0)
sorted_452, sorted_226, sorted_113, sorted_57, sorted_29 = sort_jsons(mainPath, folder_2c, resolutions)
df_452 = readin_json_merge_all(mainPath+folder_2c, sorted_452)
print(type(df_452))
"""


#get_DFs(mainPath, folder_2c, resolutions, "2c2l")
#get_DFs(mainPath, folder_3c, resolutions, "3c2l")
#get_DFs(mainPath, folder_4c, resolutions, "4c3l")
#get_DFs(mainPath, folder_6c, resolutions, "6c3l")
get_DFs(mainPath, folder_7c, resolutions, "7c3l")
get_DFs(mainPath, folder_8c, resolutions, "8c3l")