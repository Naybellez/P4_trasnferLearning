import sys
import os
sys.path.append('../../.')
from src.modelCards import get_lin_lay, Cards, return_card

from src.runCAM import run_CAM
import numpy as np
import re


def setup(GPU):
	import torch

	if GPU == 0:
		device = "cuda:0" if torch.cuda.is_available() else "cpu"
		import SimulationSettings.SettingsCAM0  as SS
	elif GPU == 1:
		device = "cuda:0" if torch.cuda.is_available() else "cpu"
		import SimulationSettings.SettingsCAM1 as SS

	print(f"CAM Setup: ", GPU, device)

	for model_name in SS.model_names:
	
		pickle_path = SS.pickle_dir+f"{model_name}/R1_{model_name}_transfer_300E/selected/"

		#modelcards = cards.modelcards
		cards = Cards()
		modelcards = cards.modelcards
		print(f"setup  modelname {model_name}")
		print(f"setup  modelname[0] {model_name[0]}")
		model_card = return_card(modelcards, key='name',targetValue=model_name)[0]
		#print(model_card)

		

		# get seeds of alreadydone training runs to avoid overwriting them
		used_seed = []
		for file in os.listdir(SS.pickle_dir):
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
		for file in os.listdir(pickle_path):
			if file[-3:] == 'pkl':
				stem = os.path.splitext(file)[0]
				seed = int(stem.rsplit("_", 2)[-2])
				if seed in used_seed:
					print(f"seed {seed} already used, skipping file {file}")
				else:
					pkl_files.append(file)
		print(len(pkl_files))
		for pickle_file in pkl_files:
			print("loopping through pkl files")
			res = np.unique(re.findall(r"\[.*?\]", pickle_file))
			print(f"res {res}")
			linlay = get_lin_lay(model_card, res[0])
			print(linlay)
			print(f"lin lay:  {linlay}")
			stem = os.path.splitext(file)[0]
			seed = int(stem.rsplit("_", 2)[-2])
			print(f"PICKLE FILE:            {pickle_file}")
			run_CAM(GPU, model_name, linlay, pickle_path, pickle_file, seed, res)
				



