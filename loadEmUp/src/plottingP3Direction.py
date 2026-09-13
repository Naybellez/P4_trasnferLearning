import numpy as np
import matplotlib
matplotlib.use('Agg')  # do this before importing pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import seaborn as sns
from src.plotting import check_save_path
import os
import re

def checkSaveName(saveloc, savename):
    #print("checkSaveName Start")
    name, ext = os.path.splitext(savename)
    match = re.search(r"_(\d+)$", name)
    if match:
        base = name[:match.start()]
        i = int(match.group(1))
        #print("checkSaveName match")
    else:
        base =  name
        i = 0
        #print("checkSaveName else")
    new_name = savename
    #print("checkSaveName while loop starting")
    if os.path.exists(os.path.join(saveloc,savename)):
        i +=1
        new_name = f"{base}_{i}"
    #print("checkSaveName while loop end")
    return new_name


def plot_confusion(predictions:list, actual:list, title:str, run_name:str,save_location =None):
    #this wasn't designed to be given a list of batches
    #print(len(predictions), len(actual))
    save_location = check_save_path(save_location)
    sns.set()
    print("Pure Preds \n ",predictions)
    
    if type(predictions[0]) != int and type(predictions[0]) != list:
        predict_list = [int(t.detach()) for t in predictions] ##
        #predict_list = [int(t.numpy()) for t in predictions]
        print("check pred type: ",predict_list[0], type(predict_list[0]))
    else:
        predict_list = predictions
        
    if type(actual[0])!= int:
        actual = [int(l.detach()) for l in actual]
    print(f"actual pre /10 : {actual}")
    print(f"pred pre /10  {predict_list}")
    actual = [int(np.round(i/10)) for i in actual]
    #print("plot_confusion ACTUAL",np.unique(actual))
    predict_list = [int(np.round(i/10)) for i in predict_list]
    #print("plot_confusion PREDICTION",np.unique(predict_list))
    actual = np.array(actual)
    predict_list = np.array(predict_list)
    #print(f"actual  {actual}")
    #print(f"preds  {predict_list}")

    
    font1 = {'family':'serif','color':'darkblue','size':14}
    font2 = {'family':'serif','color':'darkblue','size':12}
    
    #label = np.zeros(36, dtype='float32') # 360
    label = np.arange(0, 36, 1)
    disp_labels = np.arange(0, 36, 1)
    #print(f"confmatrx labels  {type(label)}   {label.shape}   {label}")

    train_epoch_matrix = confusion_matrix(actual, predict_list, labels = label)
    disp = ConfusionMatrixDisplay(train_epoch_matrix, display_labels = disp_labels)# label)
    #disp= ConfusionMatrixDisplay.from_estimator()
    #print(f"plot_conf  len label {len(label)}   len disp_labels {len(disp_labels)}")
    
    disp.plot(cmap='plasma')
    plt.title(run_name+'\n'+title, font1) #label="Accuracy Curve \n"+title, font1)
    plt.xlabel('Predicted Label', font2)
    plt.ylabel('Target Label', font2)
    if save_location != None:
        plt.savefig(save_location+'/'+'Conf_mtrx'+title+run_name+'.png', format='png', bbox_inches="tight",)
    else:
        print("Save Location Not Specified!")
    plt.show()


def plot_predictions(preds, targets, peakdists, num_samples=5, runname="", save_loc =""):
    print("Creating predicaiton plot")
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    num_samples = min(num_samples, len(preds))
    #print(num_samples)
    fig, axes = plt.subplots(nrows=num_samples, ncols=1, sharex=True,  figsize=(6.4, num_samples*2)) # *2
    #print(len(axes))
    for i in range(num_samples): # range(num_samples):
        axes[i].plot(targets[i], label="Label", color='grey', linewidth=2) #black
        axes[i].set_ylabel('Label', color='grey')
        axes[i].tick_params(axis='y', labelcolor='grey')
        axes2 = axes[i].twinx()
        axes2.plot(preds[i], label="Pred", color='red', linestyle='--')
        axes2.set_ylabel('Pred', color = 'red')
        fig.suptitle(f"Sample {i} | Target Peak : {targets[i].argmax().item()} | Pred Peak : {preds[i].argmax().item()} | PeakDist : {peakdists[i]}") 
        #if i < num_samples :
        if i != num_samples-1:
            axes[i].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)
    #axes[0].legend(loc="right")
    #plt.legend()#loc='upper left')
    savename = checkSaveName(save_loc, f"PlotPreds_"+runname) # saveloc, savename)
    plt.savefig(save_loc+savename+".jpg", dpi=100, bbox_inches="tight")#, )
    #print("plot_predictions SAVED")
    plt.show()

def plot_pred_polar(preds, targets, num_samples=5, runname="", save_loc =""):
	print("creating polar prediction plot")
	preds = preds #.detach().cpu()
	targets = targets #.detach().cpu()
	num_samples = min(num_samples, len(preds))
	print(f"num samples {num_samples}")

	fig, axes = plt.subplots(nrows=num_samples, ncols=1, figsize=(6.4, num_samples*2), subplot_kw={'projection':'polar'}) # 15, 20
	#axarr =  fig.add_subplot(projection='polar')

	preds = np.array(preds)
	preds = [p / np.sqrt(np.sum(p**2)) for p in preds] # L2 normalisation
	targets = np.array(targets)
	targets = [l / np.sqrt(np.sum(l**2)) for l in targets] # L2 normalisation
	#print(f"len reds {len(preds)} \n len labs {len(targets)}")
	for j in range(num_samples):
		index = [i for i in range(360)]
		#print(f"len index {len(index)}")
		index = np.deg2rad(index)

		axes[j].scatter(index, targets[j], alpha=0.3, c= 'dimgrey')
		axes[j].scatter(index, preds[j], alpha=0.3, c='red')

	from matplotlib.lines import Line2D
	legend_elements = [
	Line2D(
		[0], [0],
		marker = 'o',
		color ='none',
		markerfacecolor ='dimgrey', markeredgecolor='dimgrey',
		markeredgewidth=0.1,
		markersize=8,
		label='Target'),

	Line2D(
		[0],[0],
		marker='8',
		color ='none',
		markerfacecolor='red', markeredgecolor='red',
		markeredgewidth=0.1,
		markersize=8,
		label='prediction')]

	fig.legend(handles=legend_elements, bbox_to_anchor=(0.25, 0.45, 0.5, 0.5),
		frameon=True, fontsize=10, labelspacing=1, borderpad=0.001)
	save_name = checkSaveName(save_loc, f"Polar_plotPreds_{j}"+runname)
	plt.savefig(save_loc +save_name+".png", dpi=500) # , bbox_inches="tight")
	plt.show()

