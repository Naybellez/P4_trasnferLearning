# Converting pynib run to py file run

# IMPORTS
import wandb
import sys
sys.path.append('../../.')
# CUSTOM FUNCTION IMPORTS
import Dir_learning.gpr1_runs.Simulation_settings2 as SS
from Dir_learning.gpr1_runs.DL_GO2 import _go

wandb.init(mode='offline')
#wandb.login()

config = dict({'name': f"Sweep on {SS.model_name} at {SS.resolutioncard[0]['resolution']}_clip:{SS.half_ciprange}"}) #config = dict({'name': 'Sweep on 2C'})

config.update({"method": "bayes", "metric":{"goal": "minimize", "name": "t_loss"},
              "parameters": {"epochs" :{"value" : SS.epochs},
                             "batch_size": {"value": SS.batchsize},
                             "learning_rate":{"value": SS.learning_rate},
                             "loss_fn_cards": {"value":SS.loss_fn},
                             "optimiser": {"value": SS.optim},
                             "seeds": {"values": SS.seeds},
                              "half_ciprange": {"value":SS.half_ciprange},
                             "std_dev":{"value":SS.std_dev},
                             'model_cards':{"value":SS.modelcard},
                             'resolution_cards':{"value":SS.resolutioncard}
                            }})

sweep_id = wandb.sweep(sweep= config, project=SS.projectNAME)

if __name__ == '__main__':
	#wandb.agent(sweep_id, function=_go, count=20)
	count = 0
	while count <=5:
		_go()
		count += 1
