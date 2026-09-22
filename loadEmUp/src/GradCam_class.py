import torch
import os
import matplotlib.pyplot as plt
import cv2
from src.modelCards import Cards,return_card
import ast
class CAMfromPickle():
	print("Loading Model...")
	def __init__(self, model_file, dir, data_path, modelname, seed, resolution, subfolder, parentDir, device, batchsize=64):
		with open(dir+model_file, 'rb') as f:
			model_save_dict = torch.load(f)

		print("Model State Dict Found.")
		self.model_name = modelname
		self.model_state_dict = model_save_dict
		self.seed = seed
		self.resolution = ast.literal_eval(resolution[0])
		self.half_cliprange = 22
		self.std_dev = 7
		self.device = device
		self.data_path = data_path
		self.directory = dir
		self.batchsize = batchsize
		self.subfolder = subfolder
		self.parentDir = parentDir
		self.childDir = f"{self.model_name}/{self.resolution}/{self.subfolder}/"

	def get_test_set(self):
		from torch.utils.data import DataLoader
		from src.functions import ImageProcessor
		from src.dataPreProcessingP3Direction import get_data
		from src.dataloaderP3Direction import IDSWDataSetLoader3

		IP = ImageProcessor(self.device)
		xtrain,_,_,_,xtest, _ = get_data(self.seed, self.data_path)
		avlum = IP.new_luminance(xtrain)
		cards = Cards()

		resolutioncards = cards.resolutioncards
		resolution_card = return_card(resolutioncards, key='resolution', targetValue=self.resolution)[0]
		pad = resolution_card['padding']
		print(f"PAD:    {pad}")
		testds = IDSWDataSetLoader3(x=xtest, res=self.resolution,av_lum=avlum,
							   model_name=self.model_name,half_ciprange=self.half_cliprange,
							   std_dev=self.std_dev, device=self.device, pad=pad)
		self.test = DataLoader(testds, self.batchsize, shuffle=True, drop_last=True)
		print("Test Data Is Loaded")


	def fill_model(self, model):
		import torch.nn as nn
		from src.architectures import Squeeze
		print("Filling Model...")
		
		model_linears = nn.Sequential(*list(model.linear_1.children())[:-2])
		# REPLACE FC AND SOFTMAX LAYERS
		model = nn.Sequential(model.conv_layers,nn.Flatten(), Squeeze(),model_linears, nn.Linear(100, 360), nn.Softmax(dim=0))
		model_keys = set(model.state_dict().keys())
		ckpt_keys = set(self.model_state_dict.keys())
		self.model = model

		self.model.load_state_dict(self.model_state_dict)
		print("Model Filled")

	def trans_to_image(self, img, scale):
		import numpy as np
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
		from pytorch_grad_cam.utils.image import show_cam_on_image

		print(f"creating CAM figs...")

		img_names =[]
		if self.model_name == "vgg16":
			target_layer = self.model[0]
		else:
			target_layer = self.model[0]#self.model.conv_layers

		self.model.eval().to(self.device)
		preds = []
		labs = []

		for i, batch in enumerate(self.test, 0):
			#print(f"BATCH:   {type(batch)}    {len(batch)}")
			tense, label, imgB, imNorm = batch
			#print("TENSE:    ",tense.shape)
			#print("LABEL:    ",label.shape)
			label=label.to(self.device)
			prediction  = self.model.forward(tense.to(self.device))
			#print("prediction", prediction.shape)
			[preds.append(p.detach()) for p in prediction]
			[labs.append(l.detach()) for l in label]
			CAM = GradCAM(self.model, target_layer)
			targets = None
			camout = CAM(tense, targets)

			rows = min(num_samples, len(tense))
			f = plt.figure(figsize=(10, num_samples))
			gs = gridspec.GridSpec(nrows=num_samples, ncols=2, figure=f, hspace=0.4, wspace=0.3)
			for idx, img in enumerate(tense):
				img = self.trans_to_image(img, 1)
				img_names.append(img)

				visualisation = show_cam_on_image(img, camout[idx], use_rgb=True)

				if idx < num_samples:
					ax1 = f.add_subplot(gs[idx, 0])
					ax2 = f.add_subplot(gs[idx, 1])
					ax2.imshow(visualisation)
					ax1.imshow(img)
					ax1.set_xticks([])
					ax2.set_xticks([])
					ax1.set_yticks([])
					ax2.set_yticks([])

					ax1.set_ylabel(f"L {label[idx].argmax()}")
					ax2.set_ylabel(f"P {prediction[idx].argmax()}")

				f.suptitle(f"{self.model_name} Res {self.resolution}", y=1)

				if save:
					if not os.path.exists(self.parentDir+self.childDir):
						os.makedirs(self.parentDir+self.childDir)
					imageName = f"lab_{label[idx].argmax()}_pred_{prediction[idx].argmax()}"
					plt.savefig(self.parentDir+self.childDir+imageName+".png", dpi=500)
				torch.cuda.empty_cache()
