
import wandb
#import date
import time 
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18


import SimulationSettings.Settings1 as SS
from Dir_learning.modelCardsP3Direction import get_lin_lay
from Dir_learning.functions import  ImageProcessor
from Dir_learning.modelManagment import choose_model,choose_scheduler
from Dir_learning.dataPreProcessingP3Direction import get_data
from Dir_learning.dataloaderP3Direction import IDSWDataSetLoader3
from Dir_learning.fns4wandb import set_lossfn, getAcc_fromdict
from Dir_learning.loopsP3Direction import test_loop_batch, train_val_batch
from Dir_learning.fileManagment import save2csv, save2json
from Dir_learning.plotting import learning_curve, accuracy_curve
from Dir_learning.plottingP3Direction import plot_confusion

def _go(config=None):

    if len(SS.gitHASH) <1:
        print("YOU FORGET THE GIT HASH")
        return
    else:
        print('Git Hash registered')
    
    with wandb.init(mode="offline", config=config):  
        config = wandb.config

        #for model_idx, model_card in enumerate(SS.modelcards):
        #print(model_card) # debugging
        model_card = SS.modelcard
        print(model_card)
        model_name = model_card['model']
        #model_index = model_card['idx']
        dropout = model_card['dropout'] 

        for res_idx, resolution_card in enumerate(SS.resolutioncard):
            resolution = resolution_card['resolution']
            lin_lay = get_lin_lay(model_card, resolution) # returns the value of the expected size for input to fully connected layer
            seedNum = np.random.randint(len(SS.seeds))
            seed = SS.seeds[seedNum]
            loss_type = 'MSE'
            loss_fn = set_lossfn(loss_type)
            batch = SS.batchsize
        
            # SIMULATION SET OFF PRINTS # SIMULATION SET OFF PRINTS # SIMULATION SET OFF PRINTS # SIMULATION SET OFF PRINTS 
            print('Model: ', str(model_name), f" {len(SS.modelcards)}")
            print('resolution: ', str(resolution), f" idx: {res_idx} / {len(SS.resolutioncard)}")
            print('seed: ', str(seed))
            print('loss function: ', str(loss_type))
            print('Batch size: ', SS.batchsize)
            print('Training epochs: ', SS.epochs)
            print(SS.device)
            run_start_time = time.process_time()
            print('start time: ',run_start_time)
        
            epochs = SS.epochs #40
        
            IP = ImageProcessor(SS.device)
        
            #wandb.log({'gitHash':SS.gitHASH})
                    #wandb.log({'Epochs': epochs})
                    #wandb.log({'schedType':SS.scheduler_value})
        
                    # DICTIONARY TO HOLD SIMULATION SETTINGS
            save_dict = {'Run' : f"{model_name}_{resolution}",
                    'start_epoch' : 0,
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
                model = resnet18(weights=None, num_classes =360).to(SS.device)
                model_index = 100
            else:
                model = choose_model(model_name, lin_lay, dropout, SS.output_lin_lay).to(SS.device)
        
            # FOR increases DS size in training via augmentations (yaw augmentations) only create the DSL for test here, Train and Val in epoch loop
            # that will give different yaw augmentations each loop
            # if i also increase epochs, I get more unique tries for direction learning
        
            # DATALOADING # DATALOADING # DATALOADING # DATALOADING # DATALOADING # DATALOADING # DATALOADING # DATALOADING 
            x_train, _, x_val, _, x_test, y_test = get_data(seed, SS.data_path)
            av_lum = IP.new_luminance(x_train)
            train = (x_train, resolution, av_lum, model_name, SS.half_ciprange, SS.std_dev, batch)
        
            test_ds= IDSWDataSetLoader3(x_test, resolution,av_lum,model_name, SS.half_ciprange, SS.std_dev, SS.device)
            test = DataLoader(test_ds, batch_size=SS.batchsize, shuffle=True, drop_last=True) #, num_workers=2
        
            # set optimizer
            optimizer = torch.optim.Adam(model.parameters(),lr=SS.learning_rate)
            #scheduler = choose_scheduler(save_dict, optimizer)
        
            # Tell simualtion manager (WANDB) how frequently to log progress
            #wandb.watch(model, loss_fn, log='all', log_freq=2, idx = model_index)
        
            # CREATE UNIQUE SIMULATION RUN NAME 
            loop_run_name = f"{save_dict['Run']}_{resolution}_{SS.learning_rate}_{SS.scheduler_value}_{seed}_{loss_type}"
        
            # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING # TRAINING 
            model, save_dict = train_val_batch(model, train, x_val, save_dict, SS.learning_rate, loss_fn,epochs, SS.batchsize, optimizer, SS.scheduler_value, SS.device, config)
        
            # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING # TESTING 
            test_acc, test_predict_list, y_test = test_loop_batch(model,test, loss_fn, SS.batchsize, SS.device,config, runname=loop_run_name, save_loc = save_dict['save_location']) #model, model_name, X, Y, res, pad, loss_fn, device, num_classes=11
            test_predict_numerical = [p.item() for p in test_predict_list]
            y_test_numerical = [y.item() for y in y_test]
        
            # PRINTING AND SAVING # PRINTING AND SAVING # PRINTING AND SAVING # PRINTING AND SAVING # PRINTING AND SAVING # PRINTING AND SAVING 
            #wandb.log({'test_predict': test_predict_list})
            #wandb.log({'test_labels': list(y_test)})
            #save_dict.update({'test_acc': test_acc})
            save_dict['TESTAccBase'] = test_acc['BaseAcc']
            save_dict['TESTAccMSE'] = test_acc['MSE']
            save_dict['TESTAccMAE'] = test_acc['MAE']
            save_dict['TESTAccPeakDist'] = test_acc['peakDist']
            save_dict.update({'test_predict': test_predict_list})
            save_dict.update({'test_labels': list(y_test)})
            # print accuracies for each stage
            print(' \n Train Acc: ', save_dict['train_PEAKDIST'][-1])
            print(' \n Val Acc: ', save_dict['val_PEAKDIST'][-1])
            print(' \n Test Acc: ', test_acc)
        
        
            # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING # PLOTTING 
            # unpack accuracies for plotting
            #t_acc = save_dict['t_accuracy_list'] # Training
            #tbase_acc, tMSE, tMAE, t_peakdist = getAcc_fromdict(t_acc)
        
            #v_acc = save_dict['v_accuracy_list'] # Validation
            #vbase_acc, vMSE, vMAE, v_peakdist = getAcc_fromdict(v_acc)
        
            # plotting
            learning_curve(save_dict['t_loss_list'], save_dict['v_loss_list'], save_location=save_dict['save_location'],run_name=loop_run_name)
        
            accuracy_curve(save_dict['train_baseAcc'], save_dict['val_baseAcc'] ,save_location=save_dict['save_location'],run_name="Basic"+loop_run_name)
            accuracy_curve(save_dict['train_MSE'], save_dict['val_MSE'] ,save_location=save_dict['save_location'],run_name="MSE"+loop_run_name)
            accuracy_curve(save_dict['train_MAE'], save_dict['val_MAE'] ,save_location=save_dict['save_location'],run_name="MAE"+loop_run_name)
            accuracy_curve(save_dict['train_PEAKDIST'], save_dict['val_PEAKDIST'] ,save_location=save_dict['save_location'],run_name="PeakDist"+loop_run_name)
        
            plot_confusion(predictions= test_predict_numerical, actual= y_test_numerical, title = "Test Confusion matrix", run_name = loop_run_name,save_location =save_dict['save_location'])
        
            # SAVING # SAVING # SAVING # SAVING # SAVING # SAVING # SAVING # SAVING 
            save_dict.update({'run time': (time.process_time() - run_start_time)})
        
            _save_location = save_dict['save_location']
            title = save_dict['Run']
            save2json(save_dict, loop_run_name, _save_location)
            save2csv(save_dict, title, _save_location)
        
            torch.save(model.state_dict(), f"{SS.save_location}{loop_run_name}.pkl")
            torch.cuda.empty_cache()
