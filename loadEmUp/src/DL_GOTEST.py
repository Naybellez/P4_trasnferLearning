from src.GL_GOTEST import get_model, get_testData, TEST
from src.functions import ImageProcessor
from src.modelCardsP3Direction import get_lin_lay
from src.plottingP3Direction import plot_pred_polar
from src.fileManagment import save2csv

import os

# set device
#if GPU == 0:
#	device = "cuda:0" if torch.cuda.is_available() else "cpu"
#	import SimulationSettings.SettingsTest as ss
#	torch.cuda.empty_cache()
#elif GPU == 1:
#	device = "cuda:1" if torch.cuda/is_available() else "cpu"
#	import SimulationSettings.SettingsTest as ss
#	torch.cuda.empy_cache()

def run_TEST(device, ss):
	# get model
	model_card = ss['modelcard']
	print(model_card)
	model_name = model_card['model']
	resolution = ss['resolution_card']['resolution']
	linlay = get_lin_lay(model_card, resolution)
	model = get_model(device, ss['model_name'], ss['pickle'], linlay, ss['dropout'], ss['output_linlay'])

	# get data
	testLOADER = get_testData(device, ImageProcessor, ss)

	# do the test
	preds, labels = TEST(testLOADER, model, ss, device)

	if not os.path.exists(ss['save_dir']):
		os.makedirs(ss['save_dir'])

	plot_pred_polar(preds, labels, save_loc = ss['save_dir'])

	preds = [pred.to('cpu') for pred in preds]
	labels = [lab.to('cpu') for lab in labels]

	results = {'preds':preds, 'labs':labels}
	save2csv(results, 'Results', ss['save_dir'])



