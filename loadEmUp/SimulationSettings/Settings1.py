import os
import torch
from src.modelCards import Cards, return_card
import random

save_dir  = "saves/"
data_path = "/its/home/nn268/antvis/antvis/optics/NC_IDSW/"
gitHASH = " 32d139c134ef0530871dc082bae2923877f71936"

model_name = '6c3l'

pklPath = f"/its/home/nn268/antvis/antvis/transferLearning/P4_transferLearning/P1_Returns/saves/P1R/{model_name}/"

pkl_files = []
for file in os.listdir(pklPath):
	if file[-3:] == 'pkl':
		pkl_files.append(file)

epochs = 300
tv = "R1"
batchsize = 64
half_ciprange = 22 # (roughly half of 45)
std_dev = 7
learning_rate = 1e-4

output_lin_lay = 360 ###### Output labels for direction prediction specifically. 

loss_fn = ['MSE']
optim = ["adam"]
scheduler_value = "NoSched"

projectNAME = f"{tv}_{model_name}_transfer_{epochs}E"

full_path = save_dir+f"{tv}"+"/"+model_name+"/"+projectNAME+"/"
if not os.path.exists(full_path):
    os.makedirs(full_path)
save_location = full_path

