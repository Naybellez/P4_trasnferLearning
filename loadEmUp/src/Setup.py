# Converting pynib run to py file run

# IMPORTS
import wandb
import sys
sys.path.append('../../.')
# CUSTOM FUNCTION IMPORTS

#import SimulationSettings.Settings0 as SS
from src.DL_GO import run_go


def setup(GPU):
	import torch
	wandb.init(mode='offline')

	if GPU == 0:
		device = "cuda:0" if torch.cuda.is_available() else "cpu"
		import SimulationSettings.Settings0 as SS 

	elif GPU == 1:
		device = "cuda:1" if torch.cuda.is_available() else "cpu"
		import SimulationSettings.Settings1 as SS

	print("Setup: ",GPU,  device)
	print(SS.model_name)

	config = dict({'name': f"Sweep on {SS.model_name} at_clip:{SS.half_ciprange}"}) #config = dict({'name': 'Sweep on 2C'})

	config.update({"method": "bayes", "metric":{"goal": "minimize", "name": "t_loss"},
                "parameters": {"epochs" :{"value" : SS.epochs},
                                "batch_size": {"value": SS.batchsize},
                                "learning_rate":{"value": SS.learning_rate},
                                "loss_fn_cards": {"value":SS.loss_fn},
                                "optimiser": {"value": SS.optim},
                                "half_ciprange": {"value":SS.half_ciprange},                                "std_dev":{"value":SS.std_dev},
                                }})

	count = 0
	while count <= len(SS.pkl_files):
		run_go(GPU)
		count += 1

