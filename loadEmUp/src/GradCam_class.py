import torch
import os
import random
import matplotlib.pyplot as plt
import cv2
from src.loopsP3Direction import MAE_metric, MSE_metric, peak_disterr_metric2, get_roughAcc
from src.plottingP3Direction import plot_confusion, plot_predictions, plot_pred_polar
from src.fileManagment import save2json, save2csv
import sys

class CAMfromPickle():
	print("Loading Model...")
	def __init__(self, model_file, dir, data_path, modelname, seed, resolution, subfolder, parentDir, device, batchsize=64):
		with open(dir+model_file, 'rb') as f:
			model_save_dict = torch.load(f)

		print("Model State Dict Found.")
		print(f"model Save dict   : \n {list(model_save_dict)}")
		self.model_name = modelname
		self.model_state_dict = model_save_dict
		self.seed = seed
		self.resolution = resolution
		self.half_cliprange = 22
		self.std_dev = 7
		self.device = device
		self.data_path = data_path
		self.directory = dir
		self.batchsize = batchsize
		self.subfolder = subfolder
		self.parentDir = parentDir
		#self.subfolder =  f'clean/'
		#self.parentDir = "/its/home/nn268/antvis/antvis/CNN_DirectionLearning/saves/tests/CAM/"
		self.childDir = f"{self.model_name}/{self.resolution}/{self.subfolder}/"


	def get_test_set(self):
		from torch.utils.data import DataLoader
		from src.functions import ImageProcessor
		from src.dataPreProcessingP3Direction import get_data
		from src.dataloaderP3Direction import IDSWDataSetLoader3

		IP = ImageProcessor(self.device)
		xtrain,_,_,_,xtest, _ = get_data(self.seed, self.data_path)
		avlum = IP.new_luminance(xtrain)
		# To Do: Create a test DataLoader with image manipulations
		testds = IDSWDataSetLoader3(xtest, self.resolution, avlum, self.model_name, self.half_cliprange, self.std_dev, self.device)
		self.test = DataLoader(testds, self.batchsize, shuffle=True)
		print("Test Data Is Loaded")

	def fill_model(self, model):
		print("Filling Model...")
		self.model = model
		self.model.load_state_dict(self.model_state_dict)
		print("Model Filled")

	def trans_to_image(self, img, scale):
		import numpy as np
		#print(f"trans_to_image: \n {type(img)}")
		if isinstance(img, torch.Tensor):
			if len(img.shape) == 4:
				img = img.squeeze()
				img = img.permute(2, 3, 1, 0)
				img = np.array(img.cpu())*scale
			elif len(img.shape) == 3:
				img = img.squeeze()
				img = img.permute(1, 2, 0)
				img = np.array(img.cpu())*scale
			else:
				print("Image Shape Not recognised!", img.size, img.shape, len(img.shape))
		elif isinstance(img, np.ndarray):
			img = img * scale
		elif isinstance(img, str):
			img = cv2.imread(img)
			img = img * scale
		#self.img = img
		return img

	def createCAMFig(self, save=True, num_samples=5):
		import matplotlib.gridspec as gridspec

		from pytorch_grad_cam import GradCAM
		from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
		from pytorch_grad_cam.utils.image import show_cam_on_image

		img_names =[]
		if self.model_name == "vgg16":
			target_layer = self.model[0]
		else:
			target_layer = self.model.conv_layers

		self.model.eval()
		# batch enumeration
		#with torch.no_grad():
		preds = []
		labs = []


		MSE_errs = []
		MAE_errs = []
		rough_acc = []
		peakDist = []
		peakDistMean = []
		for i, batch in enumerate(self.test, 0):
			tense, label, imgB, imNorm = batch
			prediction  = self.model.forward(tense.to(self.device))
			[preds.append(p.detach()) for p in prediction]
			[labs.append(l.detach()) for l in label]
			CAM = GradCAM(self.model, target_layer)
			targets = None
			camout = CAM(tense, targets)

			rows = min(num_samples, len(tense))
			f = plt.figure(figsize=(10, num_samples))
			gs = gridspec.GridSpec(nrows=num_samples, ncols=2, figure=f, hspace=0.4, wspace=0.3)
			#print(f"len batch tense {len(tense)}")
			for idx, img in enumerate(tense):
				#print(f"len idx {len(img)}")
				#print(f"creatCAMFig : \n enumerate(tense) \n {type(img)}")
				img = self.trans_to_image(img, 1)
				#print(f"createCAMFig : \n {type(img)}")
				#print(f"tense {tense}")
				#print(f"img  {img}")
				img_names.append(img)

				visualisation = show_cam_on_image(img, camout[idx], use_rgb=True)

				if idx < num_samples:
					ax1 = f.add_subplot(gs[idx, 0])
					ax2 = f.add_subplot(gs[idx, 1])
					ax2.imshow(visualisation)
					ax1.imshow(img)
					#ax2.axis(False)
					#ax1.yaxis.set_visible(False)
					#ax1.xaxis.set_visible(False)
					#ax2.yaxis.set_visible(False)
					#ax2.xaxis.set_visible(False)
					#ax1.tick_params(left=False, bottom=False, labelleft=False, labelright=False)
					ax1.set_xticks([])
					ax2.set_xticks([])
					ax1.set_yticks([])
					ax2.set_yticks([])

					ax1.set_ylabel(f"L {label[idx].argmax()}")
					ax2.set_ylabel(f"P {prediction[idx].argmax()}")


				f.suptitle(f"{self.model_name} Res {self.resolution}", y=1)
				#plt.axis(False)

				if save:
					if not os.path.exists(self.parentDir+self.childDir):
						os.makedirs(self.parentDir+self.childDir)
					imageName = f"lab_{label[idx].argmax()}_pred_{prediction[idx].argmax()}"
					plt.savefig(self.parentDir+self.childDir+imageName+".png", dpi=500)

			test_acc = get_roughAcc(self.std_dev, label, prediction)
			rough_acc.append(test_acc)
			MSE_err = MSE_metric(prediction, label)
			MSE_errs.append(MSE_err)
			MAE_err = MAE_metric(prediction, label)
			MAE_errs.append(MAE_err)
			test_peakdist, test_meanpeakdist = peak_disterr_metric2(prediction, label)
			peakDist.append(test_peakdist)
			peakDistMean.append(test_meanpeakdist)

			plot_predictions(prediction, label, test_peakdist, runname=f"LabelPredDistributions_{i}_{idx}_Seed{self.seed}_res{int(test_acc[-1])}", save_loc=self.parentDir+self.childDir)
			plot_pred_polar(prediction.detach().cpu(), label.detach().cpu(), runname=f"Polar_LabelPredDistributions_{i}_{idx}_Seed{self.seed}_res{int(test_acc[-1])}", save_loc = self.parentDir+self.childDir)
		print("Rough Accuracy : ",rough_acc)
		print(f"Mean Peak Distances : {peakDistMean}")
		ps = [p.argmax() for p in preds]
		ls = [l.argmax() for l in labs]
		plot_confusion(ps, ls, f"ConfusionMatrix_{self.model_name}_{self.resolution}", run_name=f"Seed{self.seed}_{int(test_acc[-1])}", save_location=self.parentDir+self.childDir)
		# Saving
		savedict = {'modelanme':self.model_name,
				'testType': self.subfolder,
				'MSE':MSE_errs,
				'MAE':MAE_errs,
				'roughAcc':rough_acc,
				'PeakDists':peakDist,
				'PeakDistMean': peakDistMean,
				'Labels': labs,
				'Predictions':preds}
		save2csv(savedict, f"{self.model_name}", self.parentDir+self.childDir)
		save2json(savedict, f"{self.model_name}_Seed{self.seed}", self.parentDir+self.childDir)
