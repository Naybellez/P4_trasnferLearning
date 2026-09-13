import torch
import torch.nn as nn
from src.modelManagment import choose_model
from src.architectures import Squeeze
import sys
sys.path.append("../.")
import os
import pickle
#76E8-CACF
dirPath = "/media/noli/76E8-CACF/P1_Returns/saves/P1R/" #/its/home/nn268/antvis/antvis/transferLearning/P4_transferLearning/" # P1_Returns/saves/P1R/"
path_4c = "4c3l/"
path_3c = "3c2l/"
path_2c = "2c2l/"
path_6c = "6c3l/"
path_7c = "7c3l/"
path_8c = "8c3l/"


#filename = "P1R2_4c3l_60E_1e-44c3l_[226, 72]_0.0001_56.pkl"

def sort_files_by_res(dirPath,model_folder):
	files = []
	files_452 = []
	files_226 = []
	files_113 = []
	files_57 = []
	files_29 = []


	for file in os.listdir(dirPath+path_6c):#:
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
	print("452",len(files_452))
	print("226",len(files_226))
	print("113",len(files_113))
	print("57",len(files_57))
	print("29",len(files_29))
	return files_452, files_226, files_113, files_57, files_29
#print(files_57)

print("4c")
files_452_4c, files_226_4c, files_113_4c, files_57_4c, files_29_4c = sort_files_by_res(dirPath, path_4c)
print("3c")
files_452_3c, files_226_3c, files_113_3c, files_57_3c, files_29_3c = sort_files_by_res(dirPath, path_3c)
print("2c")
files_452_2c, files_226_2c, files_113_2c, files_57_2c, files_29_2c = sort_files_by_res(dirPath, path_2c)
print("6c")
files_452_6c, files_226_6c, files_113_6c, files_57_6c, files_29_6c = sort_files_by_res(dirPath, path_6c)
print("7c")
files_452_7c, files_226_7c, files_113_7c, files_57_7c, files_29_7c = sort_files_by_res(dirPath, path_7c)
print("8c")
files_452_8c, files_226_8c, files_113_8c, files_57_8c, files_29_8c = sort_files_by_res(dirPath, path_8c)


model = choose_model("6c3l", lin_lay=71680,dropout=0.2) #141056
print(model)
print("model loaded")
with open(dirPath+path_6c+files_226_6c[3], 'rb') as f: #
	checkpoint = pickle.load(f) # 

print("checkoint loaded")


#	checkpoint = torch.load(f, weights_only=False)
#checkpoint = torch.load(dirPath+files_226[1],#
#		map_location=lambda storage, loc: storage,
#		weights_only=False)
#checkpoint = pickle.load(dirPath+files_226[1])
#print(type(checkpoint))
#print(checkpoint.keys() if isinstance(checkpoint, dict) else "not a dict")#


"""model.load_state_dict(checkpoint['model.state_dict'])
print("Weights loaded")
#print(model)
model_linears = torch.nn.Sequential(*list(model.linear_1.children())[:-2])
#print(model)
model = nn.Sequential(model.conv_layers,nn.Flatten(), Squeeze(), model_linears, nn.Linear(100, 360), nn.Softmax(dim=0))
#print(model)
import re
import numpy as np
s = files_226_6c[0]
print(s)
print(int(re.search(r'\d+', s[40:43]).group()))
print(np.unique(re.findall(r"\[(.*?)\]",s)))

"""