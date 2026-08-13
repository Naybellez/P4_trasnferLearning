import sys
sys.path.append('../../.')

def setup(GPU):
	import torch

	if GPU == 0:
		device = "cuda:0" if torch.cuda.is_available() else "cpu"
		import SimulationSettings.SettingsCAM0  as SS
	elif GPU == 1:
		device = "cuda:1" if torch.cua.is_available() else "cpu"
		import SimulationSettings.SettingsCAM1 as SS

	print(f"CAM Setup: ", GPU, device)

	#epochs =300
	#lr ="1e-4"

	cards = Cards()
	#resolutions = [[226, 72],[113, 36],[57, 18],[29, 9],[15, 5], [8, 3]]

	#model_names = ["2c2l","3c2l","4c3l","6c3l","7c3l", "8c3l", "10c4l"]
	# "VGG16", "resnet18"]

	for model_name in SS.model_names:
		for res in SS.resolutions:

		pickle_path = f"/its/home/nn268/antvis/antvis/CNN_DirectionLearning/saves/{res}/{model_name}/"
		subfolder = "testingAll/clean/"
		parentDir = "/its/home/nn268/antvis/antvis/CNN_DirectionLearning/saves/tests/CAM/"

		data_path = "/its/home/nn268/antvis/antvis/optics/NC_IDSW/"


		modelcards = cards.modelcards

		modelcard = return_card(modelcards, key='name', targetValue=model_name)[0]
		linlay = get_lin_lay(modelcard, res)
		model = choose_model(model_name, linlay, 0, 360).to(device)

		pkl_seeds = get_seeds(resolution=res, modelname=model_name, picklePath=picklePath)

		for seed in pkl_seeds:
			if model_name != "10c4l":
				pickle_file = f"{model_name}_{SS.epochs}E_{SS.lr}_ADAM_{res}_{model_name}_{res}_{res}_0.0001_NoSched_{seed}_MSE.pkl"
			else:
				pickle_file = f"TESTING_{model_name}_{epochs}_{res}_{res}_0.0001_NoSched_{seed}_MSE.pkl"
			CAM = CAMfromPickle(model_file=pickle_file, dir=pickle_path, data_path = data_path, model_name=model_name, seed=seed, resolution=res, subfolder=subfolder, parentDir=parentDir, device=device)
			CAM.get_test_set()
			CAM.fill_model(model)
			CAM.createCAMFig()



