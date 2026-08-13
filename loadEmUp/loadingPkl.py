import torch
import torch.nn as nn
from src.modelManagment import choose_model
from src.architectures import Squeeze
import sys
sys.path.append("../.")
import os
import pickle

dirPath = "/its/home/nn268/antvis/antvis/transferLearning/P4_transferLearning/P1_Returns/saves/P1R/"
path_4c = "4c3l/"
path_3c = "3c2l/"
path_2c = "2c2l/"
path_6c = "6c3l/"
path_7c = "7c3l/"
path_8c = "8c3l/"


#filename = "P1R2_4c3l_60E_1e-44c3l_[226, 72]_0.0001_56.pkl"
files = []
files_452 = []
files_226 = []
files_113 = []
files_57 = []
files_29 = []

for file in os.listdir(dirPath+path_4c):
	if file[-3:] == 'pkl':
		#print(file[23:31])
		if file[23:33] == '[452, 144]':
			files_452.append(file)
		elif file[23:32] == '[226, 72]':
			files_226.append(file)
		elif file[23:32] == '[113, 36]':
			files_113.append(file)
		elif file[23:31] == '[57, 18]':
			files_57.append(file)
		elif file[23:30] == '[29, 9]':
			files_29.append(file)
		else:
			files.append(file)
#print(files_57)



model = choose_model("4c3l", lin_lay=141056,dropout=0.2)
with open(dirPath+path_4c+files_226[2], 'rb') as f:
	checkpoint = pickle.load(f) # torch.load(f)
#checkpoint = torch.load(dirPath+files_226[1],
#		map_location=lambda storage, loc: storage,
#		weights_only=False)
#checkpoint = pickle.load(dirPath+files_226[1])
#print(type(checkpoint))
#print(checkpoint.keys() if isinstance(checkpoint, dict) else "not a dict")
model.load_state_dict(checkpoint['model.state_dict'])
#print(model)
model_linears = torch.nn.Sequential(*list(model.linear_1.children())[:-2])
#print(model)
model = nn.Sequential(model.conv_layers,nn.Flatten(), Squeeze(), model_linears, nn.Linear(100, 360), nn.Softmax(dim=0))
print(model)
import re
import numpy as np
s = files_226[0]
print(s)
print(int(re.search(r'\d+', s[40:43]).group()))
print(np.unique(re.findall(r"\[(.*?)\]",s)))

