import os
from src.modelCards import Cards, return_card
import random


save_dir  = "/media/noli/Expansion/P4_saves/"
data_path ="/home/noli/Documents/Nay/NC_IDSW/N2CenterR1_SCenter/IDSW_ds2324_N/" # "/home/noli/Documents/Nay/NC_IDSW/"#"/its/home/nn268/antvis/antvis/optics/NC_IDSW/"
gitHASH = " 32d139c134ef0530871dc082bae2923877f71936"

model_name = '3c2l'

pklPath = f"/media/noli/Expansion/P1_saves/P1R/{model_name}/selectedPKLs/" #f"/its/home/nn268/antvis/antvis/transferLearning/P4_transferLearning/P1_Returns/saves/P1R/{model_name}/"

epochs = 300
tv = "R1"
learning_rate = 1e-4

batchsize = 64
half_ciprange = 22 # (roughly half of 45)
std_dev = 7

output_lin_lay = 360 ###### Output labels for direction prediction specifically. 

loss_fn = ['MSE']
optim = ["adam"]
scheduler_value = "NoSched"

projectNAME =f"{tv}_{model_name}_transfer_{epochs}E"


# get seeds of alreadydone training runs to avoid overwriting them
used_seed = []
for file in os.listdir(save_dir):
    if file.endswith(".pkl"):
        stem = os.path.splitext(file)[0]
        try:
            seed = int(stem.rsplit("_", 2)[-2])
            used_seed.append(seed)
        except (IndexError, ValueError):
            pass

# check files of input model folder and skip those with already used seeds
print(f"modelname  {model_name}")
pkl_files = []
for file in os.listdir(pklPath):
	if file[-3:] == 'pkl':
		stem = os.path.splitext(file)[0]
		seed = int(stem.rsplit("_", 2)[-2])
		if seed in used_seed:
			print(f"seed {seed} already used, skipping file {file}")
		else:
			pkl_files.append(file)



full_path = save_dir+f"{tv}"+"/"+model_name+"/"+projectNAME+"/"
if not os.path.exists(full_path):
    os.makedirs(full_path)
save_location = full_path
