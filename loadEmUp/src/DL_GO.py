
import wandb
#import date
import time 
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import re
#from src.modelCardsP3Direction import Cards, return_card
from src.modelCards import Cards, return_card
import pickle
#import Simulation_settings as SS
from src.architectures import Squeeze
from src.modelCardsP3Direction import get_lin_lay
from src.functions import  ImageProcessor
from src.modelManagment import choose_model,choose_scheduler
from src.dataPreProcessingP3Direction import get_data
from src.dataloaderP3Direction import IDSWDataSetLoader3
from src.fns4wandb import set_lossfn, getAcc_fromdict
from src.loopsP3Direction import test_loop_batch, train_val_batch
from src.fileManagment import save2csv, save2json
from src.plotting import learning_curve, accuracy_curve
from src.plottingP3Direction import plot_confusion

import ast

def run_go(GPU):
	if GPU == 0:
		device = "cuda:0" if torch.cuda.is_available() else "cpu"
		import SimulationSettings.Settings0 as SS 
		torch.cuda.empty_cache()

	elif GPU == 1:
		device = "cuda:1" if torch.cuda.is_available() else "cpu"
		import SimulationSettings.Settings1 as SS
		torch.cuda.empty_cache()

	#print(f"run_go:  {GPU}  {device}")
	torch.cuda.empty_cache()

	def _go(config=None):

		if len(SS.gitHASH) <1:
		    print("YOU FORGET THE GIT HASH")
		    return
		else:
		    print('Git Hash registered')

		#with wandb.init(mode="offline", config=config):  
		#config = wandb.config

		model_name = SS.model_name
		print(model_name)
		cards = Cards()
		modelcards = cards.modelcards
		model_card = return_card(modelcards, key='name',targetValue=model_name)[0]

		print(model_card)
		model_name = model_card['model']
		dropout = model_card['dropout']
		print(len(SS.pkl_files))
		seedNum = np.random.randint(len(SS.pkl_files))
		pkl_f = SS.pkl_files[seedNum]

		print(pkl_f)
		print(type(pkl_f))
		print( pkl_f[-15:-11])
		print(int(re.search(r'\d+', pkl_f[-15:-11]).group()))
		seed = int(re.search(r'\d+', pkl_f[-15:-11]).group()) #SS.seeds[seedNum] # 40:43
		res = np.unique(re.findall(r"\[.*?\]", pkl_f))
		res = ast.literal_eval(res[1])
		print(f"SEED: {seed} {type(seed)} | RES:  {res} {type(res)}")

		resolutioncards = cards.resolutioncards
		resolution_card = return_card(resolutioncards, key='resolution', targetValue=res)[0]
		#print(f"resolutionCARD   {resolution_card}")
		resolution = resolution_card['resolution']
		pad = resolution_card['padding']
		print(f"PAD:    {pad}")
		#print(f"resolution     {resolution}")
		#if res == resolution:
		#	print("RESOLUTION MATCH")
		#else:
		#	print("Res MisMatch")
		lin_lay = get_lin_lay(model_card, resolution) # returns the value of the expected size for input to fully connected layer
		loss_type = 'MSE'
		loss_fn = set_lossfn(loss_type)
		batch = SS.batchsize

		# SIMULATION SET OFF PRINTS # SIMULATION SET OFF PRINTS # SIMULATION SET OFF PRINTS # SIMULATION SET OFF PRINTS 
		print('Model: ', str(model_name))
		print('resolution: ', str(resolution))
		print('seed: ', str(seed))
		print('Batch size: ', SS.batchsize)
		print('Training epochs: ', SS.epochs)
		print(device)
		run_start_time = time.process_time()
		print('start time: ',run_start_time)

		epochs = SS.epochs #40

		IP = ImageProcessor('cpu')

		# DICTIONARY TO HOLD SIMULATION SETTINGS
		save_dict = {'Run' : f"{model_name}_{resolution}",
			'start_epoch' : 60,
			'Current_Epoch': 0,
			'save_location' : SS.save_location,
			'scheduler': SS.scheduler_value,
			'gitHASH':str(SS.gitHASH),
			'model_name': str(model_name),
			'loss_fn': str(loss_type),
			'lr': str(SS.learning_rate),
			'resolution': str(resolution),
			'seed': str(seed),
			'lin_lay': int(lin_lay)}


		# SELECTING THE MODEL BASED ON MODEL_NAME
		if model_name == 'resnet18':
			model = resnet18(weights=None, num_classes =360)#.to(device)
			model_index = 100
		else:
			model = choose_model(model_name, lin_lay, dropout)
			torch.cuda.empty_cache()
		# LOAD IN MODEL STATE DICT
		#try :
		#	print("try except start")
		print(pkl_f)
		with open(SS.pklPath+pkl_f, 'rb') as f:
			print("file opened")
			#checkpoint = pickle.load(f)
			checkpoint = torch.load(f, map_location=device, weights_only=True)
		print(checkpoint.keys())
		model.load_state_dict(checkpoint) # ['model.state_dict']
		
		#except:
		#	print(f"BAD FILE \n BAD FILE:  {f} \n {pkl_f} \n BAD FILE")

			
		# RM LAST FC LAYER
		model_linears = nn.Sequential(*list(model.linear_1.children())[:-2])
		# REPLACE FC AND SOFTMAX LAYERS
		model = nn.Sequential(model.conv_layers,nn.Flatten(), Squeeze(),model_linears, nn.Linear(100, 360), nn.Softmax(dim=0))
		model.to(device)
		#print("Model:  ",model)
		# FOR increases DS size in training via augmentations (yaw augmentations) only create the DSL for test here, Train and Val in epoch loop
		# that will give different yaw augmentations each loop
		# if i also increase epochs, I get more unique tries for direction learning

		# DATALOADING # DATALOADING # DATALOADING # DATALOADING # DATALOADING # DATALOADING # DATALOADING # DATALOADING 
		print("Model Loaded.  \n Loading Data...")
		print(SS.data_path)
		
		x_train, _, x_val, _, x_test, y_test = get_data(seed, SS.data_path)
		print("LEN x train:  ",len(x_train))
		av_lum = IP.new_luminance(x_train)
		train = (x_train, resolution, pad, av_lum, model_name, SS.half_ciprange, SS.std_dev, batch)

		test_ds= IDSWDataSetLoader3(x_test, resolution,pad,av_lum,model_name, SS.half_ciprange, SS.std_dev, device)
		print("LEN test custom loader : ", len(test_ds))
		test = DataLoader(test_ds, batch_size=SS.batchsize, shuffle=True, drop_last=True) #, num_workers=2
		print("LEN test loader : ",len(test))
		optimizer = torch.optim.Adam(model.parameters(),lr=SS.learning_rate)

		# CREATE UNIQUE SIMULATION RUN NAME 
		loop_run_name = f"{save_dict['Run']}_{resolution}_{SS.learning_rate}_{SS.scheduler_value}_{seed}_{loss_type}"

		# TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING 
		model, save_dict = train_val_batch(model, train, x_val, save_dict,  loss_fn,epochs, optimizer, device, config, SS)
		print("Trained...")
		# TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING 
		test_acc, test_predict_list, y_test = test_loop_batch(model,test, loss_fn,  device,config, SS, runname=loop_run_name, save_loc = save_dict['save_location']) #model, model_name, X, Y, res, pad, loss_fn, device, num_classes=11
		#test_predict_numerical = [p.item() for p in test_predict_list]
		#y_test_numerical = [y.item() for y in y_test]

		save_dict['TESTAccBase'] = test_acc['BaseAcc']
		save_dict['TESTAccMSE'] = test_acc['MSE']
		save_dict['TESTAccMAE'] = test_acc['MAE']
		save_dict['TESTAccPeakDist'] = test_acc['peakDist']
		save_dict.update({'test_predict': test_predict_list})
		save_dict.update({'test_labels': list(y_test)})
		# print accuracies for each stage
		#print(' \n Train Acc: ', save_dict['train_PEAKDIST'][-1])
		#print(' \n Val Acc: ', save_dict['val_PEAKDIST'][-1])
		print(' \n Test Acc: ', test_acc)
		print(f"len test predict  {len(save_dict['test_predict'])}")
		#print(save_dict['test_predict'][0])
		print(f"len labels    {len(save_dict['test_labels'])}")
		#print(save_dict['test_labels'])
		#import pandas as pd
		#save_dict = pd.DataFrame(save_dict)
		#save_dict = save_dict.explode('test_predict','test_labels')
		# plotting
		print(f"t loss    {save_dict['t_loss_list'][:5]}   {type(save_dict['t_loss_list'])}")
		learning_curve(save_dict['t_loss_list'], save_dict['v_loss_list'], save_location=save_dict['save_location'],run_name=loop_run_name)

		accuracy_curve(save_dict['train_baseAcc'], save_dict['val_baseAcc'] ,save_location=save_dict['save_location'],run_name="Basic"+loop_run_name)
		accuracy_curve(save_dict['train_MSE'], save_dict['val_MSE'] ,save_location=save_dict['save_location'],run_name="MSE"+loop_run_name)
		accuracy_curve(save_dict['train_MAE'], save_dict['val_MAE'] ,save_location=save_dict['save_location'],run_name="MAE"+loop_run_name)
		accuracy_curve(save_dict['train_PEAKDIST'], save_dict['val_PEAKDIST'] ,save_location=save_dict['save_location'],run_name="PeakDist"+loop_run_name)

		#plot_confusion(predictions= test_predict_numerical, actual= y_test_numerical, title = "Test Confusion matrix", run_name = loop_run_name,save_location =save_dict['save_location'])

		# SAVING # SAVING # SAVING # SAVING # SAVING # SAVING # SAVING # SAVING 
		save_dict.update({'run time': (time.process_time() - run_start_time)})

		_save_location = save_dict['save_location']
		title = save_dict['Run']
		save2json(save_dict, loop_run_name, _save_location)
		print("saved to JSON")
		#save2csv(save_dict, title, _save_location)
		#print("Saved to csv")
		model.cpu()
		torch.save(model.state_dict(), f"{SS.save_location}{loop_run_name}.pkl")
		del model
		torch.cuda.empty_cache()
	_go()
