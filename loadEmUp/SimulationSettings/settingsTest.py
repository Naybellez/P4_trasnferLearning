import os
import random
from src.modelCardsP3Direction import Cards, return_card
from src.modelManagment import get_seeds
from src.Test_pkls import SampleTestPkls
import re
import numpy as np
import ast

def get_config(k):
	pkl = SampleTestPkls[k]
	dirPath = "/its/home/nn268/antvis/antvis/CNN_DirectionLearning/"
	s_path = "/its/home/nn268/antvis/antvis/CNN_DirectionLearning/saves/"

	full_path=dirPath+pkl #+"/"+pkl
	#print(f"Full Path:   {full_path}")
	#print(os.path.isfile(full_path))
	model_name = None
	modelnames = ["2c2l", "3c2l", "4c3l", "6c3l", "7c3l", "8c3l"]
	mn = pkl[19:30]
	print(mn)
	for m in modelnames:
		print(f"m  : {m}")
		if m in mn:
			print(f"{mn} in {m}")
			model_name = m
	print("New Model Name:     ", model_name)
	print("ModelName: ",model_name)
	tv = list(set(re.findall(r"\[(.*?)\]", pkl)))
	tv[0] = "["+tv[0]+"]"
	tv = ast.literal_eval(tv[0])
	#print(f"tv    1    {tv}    {type(tv)}")
	pkl_seed = int(re.search(r'\d+', pkl[-12:]).group())
	print(f"Model name    {model_name}, {type(model_name)}")
	#print(f"tv        {tv}     {type(tv)}")
	#if not os.path.exists(full_path):
	#	os.makedirs(full_path)
	save_location=full_path

	cards = Cards()
	modelcards =cards.modelcards
	if model_name != "resnet18":
		modelcard = return_card(modelcards, key='name', targetValue=model_name)[0]
		print(f"Model Card :\n {type(modelcard)} \n {modelcard}")

	resolutioncards = cards.resolutioncards
	print(type(tv), tv)
	print("RESCARD",resolutioncards[0]['resolution'],type(resolutioncards[0]['resolution']))
	resolution_card = return_card(resolutioncards, key = "resolution",targetValue = tv)[0]
	print(f"Resolution Card : \n{type(resolution_card)} \n  {resolution_card}")
	lin_lay = cards.modname2linlay(modelcard['name'], resolution_card['resolution'])
	dropout = modelcard['dropout']

	settings_dict = {
	"data_path": "/its/home/nn268/antvis/antvis/optics/NC_IDSW/",


	"epochs": 300,
	"lr": "1e-4",
	"batchsize": 64,

	"half_cliprange": 22,
	"std_dev": 7,
	"output_linlay": 360,

	"save_dir": s_path+f"/tests3/{tv}/{model_name}/",

	"pickle_path": f"{tv}/{model_name}/pkl/",
	"pickle":full_path, 
	"projectNAME": f"Test_{model_name}_{tv}",
	"resolution_card": resolution_card,
	"res": tv,
	"modelcard": modelcard,
	"lin_lay" : lin_lay,
	"dropout": dropout,
	"model_name":model_name,
	"seed": pkl_seed,
	}

	return settings_dict

#index = random.randint(0, 3)



