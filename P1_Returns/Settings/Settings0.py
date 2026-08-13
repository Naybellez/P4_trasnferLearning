import random
import os
from Dir_learning.P4_transferLearning.P1_Returns.src.modelCards import Cards, return_card


save_dir = "C:/Users/frogl/LecunBackup_repo/AntvisBackup/Dir_learning/P4_transferLearning/P1_Returns/saves/"
datapath = "/its/home/nn268/antvis/antvis/optics/AugmentedDS_IDSW/"

model_name = "2c2l"
epochs = 60
learning_rate = 1e-4
batchsize = 64
output_linlay = 11 
loss_fn = ['MSE']
optim = 'adam'

project_name = f"P1R_{model_name}_{epochs}E_ 1e-4_{optim}"
full_path = save_dir+f"P1R/{model_name}/{project_name}"
if not os.path.exists(full_path):
    os.makedirs(full_path)
save_location = full_path




seeds = []
while len(seeds) <= 10:
    seeds.append(random.randint(1000))

cards = Cards()
modelcards = cards.modelcards
if model_name != 'resnet18':
    modelcard = return_card(modelcards, key='name', targetValue=model_name)[0]
    print(f" Model Card:  {modelcard}")

resolutioncards = cards.resolutioncards
resolutioncard_452 = return_card(resolutioncards, key='resolution', targetValue=[452, 144])[0]
resolutioncard_226 = return_card(resolutioncards, key='resolution', targetValue=[226, 72])[0]
resolutioncard_113 = return_card(resolutioncards, key='resolution', targetValue=[113, 36])[0]
resolutioncard_57 = return_card(resolutioncards, key='resolution', targetValue=[57, 18])[0]
resolutioncard_29 = return_card(resolutioncards, key='resolution', targetValue=[29, 9])[0]

all_recards = [resolutioncard_452,resolutioncard_226,resolutioncard_113,resolutioncard_57,resolutioncard_29]