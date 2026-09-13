import torch
import torch.nn as nn
from torchvision.models import vgg16, resnet18
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import numpy as np
import cv2
from datetime import date
from tqdm import tqdm
import collections
import time
import random

import csv
import json
import pickle
import os
import sys
sys.path.append('../.')
#
from src.functions import get_data, import_imagedata, ImageProcessor, label_oh_tf, IDSWDataSetLoader2
from loadEmUp.src.fns4wandb import set_lossfn
#from src.architectures import sevennet, smallnet1, smallnet2, smallnet3, PrintLayer
from src.loop_fns import loop, train_val_batch, test_loop_batch#, loop_batch, test_loop_batch
from loadEmUp.src.plotting import learning_curve, accuracy_curve, plot_confusion
from src.modelManagment import choose_model
from src.fileManagment import save2csv,save2json
from src.modelCards import get_lin_lay

def run_p1r(GPU):
    if GPU == 0:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        import Settings.Settings0 as SS
    elif GPU == 1:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        import Settings.Settings1 as SS
    #print(f"run_go:  {GPU}    {device}")
    torch.cuda.empty_cache()

    def _go():
        
        model_card = SS.modelcard
        print(model_card)
        model_name = SS.modelcard['model']
        dropout = model_card['dropout']
        seednum = np.random.randint(len(SS.seeds))
        seed = int(SS.seeds[seednum])

        for res_idx, resolution_card in enumerate(SS.all_recards):
            # local references to settings
            lin_lay = get_lin_lay(model_card, resolution_card['resolution'])
            
            pad = resolution_card['padding']
            # print simulation settings
            print(f"MODEL  {model_name}")
            print(f"RESOLUTION     {resolution_card['resolution']}")
            print(f"SEED     {seed}")
            print(f"LOSS FN     {SS.loss_fn}")
            print(f"BATCH SIZE    {SS.batchsize}")
            print(f"TRAINING EPOCHS      {SS.epochs}")
            # initiate an Image Processor instance 
            IP = ImageProcessor(device)
            # Dicitonary to hold simulation settings/ configs
            save_dict = {'Run': f"{model_name}_{resolution_card['resolution']}",
                         'start_epoch': 0,
                         'CurrentEpoch': 0,
                         'save_location': str(SS.save_location),
                         'model_name':str(model_name),
                         'loss_fn': SS.loss_fn,
                         'scheduler': "NoSched",
                         'lr': str(SS.learning_rate),
                        'resolution': str(resolution_card['resolution']),
                        'seed':str(seed),
                        'lin_lay':int(lin_lay)}#
            # Selecting model based on model name
            if model_name == 'resnet18':
                model = resnet18(weights=None, num_classes= SS.output_linlay).to(device)
                model_indec = 100
            else:
                model = choose_model(model_name, lin_lay, dropout, SS.output_linlay).to(device)
            torch.cuda.empty_cache()

            # data
            x_train, y_train, x_val, y_val, x_test, y_test = get_data(seed, SS.datapath)
            print(f"Len Xtrain   {len(x_train)}")
            av_lum = IP.new_luminance(x_train)
            train_ds = IDSWDataSetLoader2(x_train, y_train, resolution_card['resolution'], pad, av_lum, model_name, device)
            train = DataLoader(train_ds, batch_size=SS.batchsize, shuffle=True, drop_last=True)
            val_ds = IDSWDataSetLoader2(x_val, y_val, resolution_card['resolution'], pad, av_lum, model_name, device)
            val = DataLoader(val_ds, batch_size=SS.batchsize, shuffle=True, drop_last=True)
            test_ds = IDSWDataSetLoader2(x_test, y_test, resolution_card['resolution'], pad, av_lum, model_name, device)
            test = DataLoader(test_ds, batch_size=SS.batchsize, shuffle=True, drop_last=True)
            print("After data loading - Current allocated memory (GB):", torch.cuda.memory_allocated() / 1024 ** 3)

            loss_fn = set_lossfn(SS.loss_fn)
            optimizer = torch.optim.Adam(model.parameters(), lr=SS.learning_rate)
            loop_run_name = f"{save_dict['Run']}_{resolution_card['resolution']}_{SS.learning_rate}_{seed}_{SS.loss_fn}"
            # TRAINING
            print("Training...")
            model, save_dict = train_val_batch(model, train, val, loop_run_name, save_dict, SS.learning_rate, loss_fn, SS.epochs, SS.batchsize, optimizer, scheduler_value=None, device=device)
            #TESTING
            print("Testing...")
            test_acc,test_predict_list, y_test = test_loop_batch(model,test, loss_fn, SS.batchsize, device)
            save_dict.update({'test_acc': test_acc})
            save_dict.update({'test_predict': test_predict_list})
            save_dict.update({'test_labels': list(y_test)})

            
            learning_curve(save_dict['t_loss_list'], save_dict['v_loss_list'], save_location=save_dict['save_location'],run_name=loop_run_name)
            accuracy_curve(save_dict['t_accuracy_list'], save_dict['v_accuracy_list'],save_location=save_dict['save_location'],run_name=loop_run_name)
            test_predict_list=[pred for pred in test_predict_list]
            #plot_confusion(predictions= test_predict_list, actual= y_test, title = "Test Confusion matrix", run_name = loop_run_name,save_location =save_dict['save_location'])

            diction = {}
            d = date.today()
            d=str(d)
            diction.update({'Date':d})
            #diction.update({'gitHASH':str(gitHASH)})
            diction.update({'model_name': str(model_name)})
            diction.update({'loss_fn': str(SS.loss_fn)})
            diction.update({'lr': str(SS.learning_rate)})
            #diction.update({'wd': str(wd_card)})
            
            diction.update({'seed': str(seed)})
            diction.update({'resolution': str(resolution_card['resolution'])})
            diction.update({'pad': int(pad)})
            diction.update({'lin_lay': int(lin_lay)})
            #diction.update({'run time': (time.process_time() - run_start_time)})
            diction.update(save_dict)
            
            save_location = SS.save_location
            title = save_dict['Run']
            save2json(diction, loop_run_name, save_location)
            save2csv(diction, title, save_location)

            """diction['model.state_dict'] = model.state_dict() #to('cpu').

            with open(f"{save_location}{loop_run_name}.pkl", 'wb+') as f:
                pickle.dump(diction, f)"""
            torch.save(model.state_dict(), f"{SS.save_location}{loop_run_name}.pkl")
            del model
            torch.cuda.empty_cache()

    _go()

