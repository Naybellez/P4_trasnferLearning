# imports external
import torch
from torch.utils.data import DataLoader
# imports internal
from src.modelCardsP3Direction import get_lin_lay
from src.functions import ImageProcessor
from src.modelManagment import choose_model, choose_scheduler
from src.dataPreProcessingP3Direction import get_data
from src.dataloaderP3Direction import IDSWDataSetLoader3
from src.fns4wandb import set_lossfn, getAcc_fromdict
from src.loopsP3Direction import test_loop_batch, MSE_metric, MAE_metric, peak_disterr_metric2, get_roughAcc
#from src.fileManagment import learning_curve, accuracy_curve
from src.plottingP3Direction import plot_confusion


# load model from pkl
def get_model(device, model_name, pickle, lin_lay, dropout, output_linlay):
	model = choose_model(model_name, lin_lay, dropout, output_linlay)
	model_weights = torch.load(pickle)#['model_state_dict'])
	print(f"\n model Weights \n :  {list(model_weights)}\n")
	model.load_state_dict(model_weights)
	print(model)
	return model.to(device)

# Load Data
def get_testData(device, ImageProcessor, ss):
	IP = ImageProcessor(device)
	xtrain,_,_,_, x_test,_ = get_data(ss['seed'], ss['data_path'])
	av_lum = IP.new_luminance(xtrain)
	test_ds = IDSWDataSetLoader3(x_test, ss['res'], av_lum, ss['model_name'], ss['half_cliprange'], ss['std_dev'], device)
	test = DataLoader(test_ds, batch_size=ss['batchsize'], shuffle =True, drop_last =True)
	return test

# test loop
def TEST(data, model, ss, device):

	model = model.eval()
	predict_list = []
	label_list = []
	peakdists = []
	peakdistMeans = []
	baseacc_list = []
	MSE_list = []
	MAE_list = []

	with torch.no_grad():
		for i, batch in enumerate(data, 0):
			tense, label, imgB, imNorm =  batch
			prediction = model.forward(tense.to(device))
	[predict_list.append(pred.detach().to('cpu'))for pred in prediction]
	[label_list.append(lab.detach().to('cpu'))for lab in label]

	tacc = get_roughAcc(ss['std_dev'], label, prediction)
	MSE_list.append(MSE_metric(prediction.to('cpu'), label.to('cpu')))
	MAE_list.append(MAE_metric(prediction.to('cpu'), label.to('cpu')))
	peakdist, peakdMean = peak_disterr_metric2(prediction.to('cpu'), label.to('cpu'))
	peakdists.append(peakdist)
	peakdistMeans.append(peakdMean)
	print(f"\n Accuracy within 45degree catchment. \n tries, correct, percentage : \n {tacc}")
	print(f"\n Average Peak Distance : \n {peakdMean}")
	return predict_list, label_list


# plots 
