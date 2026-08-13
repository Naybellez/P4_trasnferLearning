# imports
import sys
sys.path.append("../../.")

# Imports Custom

from src.DL_GOTEST import run_TEST
from src.Test_pkls import SampleTestPkls

def testup(GPU):
	import torch
	if GPU == 0:
		device = "cuda:0" if torch.cuda.is_available() else "cpu"
		#import SimulationSettings.SettingsTest as ss
	elif GPU == 1:
		device = "cuda:1" if torch.cuda.is_available() else "cpu"
		#import  SimulationSettings.SettingsTest as ss

	
	print(f"testup {GPU} {device}")
	from src.Test_pkls import SampleTestPkls
	for k in range(38, len(SampleTestPkls)):
		from SimulationSettings.SettingsTest import get_config
		ss = get_config(k)
		run_TEST(GPU, ss)
