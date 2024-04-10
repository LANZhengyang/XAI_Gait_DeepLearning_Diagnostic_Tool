# Runing the XAI methods LIME, DeepShap and Integrated Gradients for the best trained ResNet model of TDvsCPu.
# Computing XAI all at once requires a lot of memory for a large number of inputs (LIME and Integrated Gradients style in this file). 
# Computing it multiple times separately can alleviate the memory shortage problem. (DeepShap style in this file)

import numpy as np
from captum.attr import visualization as viz
from captum.attr import Lime, LimeBase
from captum.attr import IntegratedGradients
from captum.attr import DeepLiftShap
import math

from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function

import torch
import pytorch_lightning as pl

from utils.utils import *
import matplotlib.pyplot as plt

import os


# Load dataset
dataset = "archives/AQM/TDvsPC_new_no_shuffle/"
d_file_list = ['CP.npz','TD.npz']
x_list, nb_classes = load_dataset_v1(dataset, d_file_list, channel_first=True,flatten=False)

# Load subject cycle table and split the dataset to training, validation and test set
idx_file_list = ['DataCP - By Subject.csv','DataTD - By Subject.csv']
x_train, x_test, y_train, y_test, x_val, y_val, cycle_end_idx, X_d, y_d = generate_data_for_train(x_list, idx_file_list,order=[1,0])


filePath='./model/new_train_lrR_ET_TDvsCPu_full_5_original_weight_ResNet/'

# Find the model with best accuracy
model_list = os.listdir(filePath)
for idx_tmp,i in enumerate(model_list):
    if not i.startswith('model'):
        model_list.remove(i)   
model_best = ''
accuracy_best = 0
for i in model_list:
    classifer = init_model(model_name='ResNet').load(filePath+i)
    accuracy_i = accuracy_score(y_d,classifer.predict(X_d))
    if accuracy_i > accuracy_best:
        accuracy_best = accuracy_i
        model_best = i
print(accuracy_best)
print(model_best)

# Used the model with best accuracy to use XAI
classifer = init_model(model_name='ResNet').load(filePath+model_best)

# Set the distance function for LIME
exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=100)

# Set the parameter for LIME
lr_lime = Lime(
    classifer.model, 
    interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
    similarity_func=exp_eucl_distance
)


# Evaluate the class 0 (TD) and class 1 (CPu) in the test set by LIME and save
target = 0
attrs_0 = lr_lime.attribute(
    torch.from_numpy(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)][y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)]==target]).to(torch.float32),
    target=target,
    n_samples=5000,
    perturbations_per_eval=200,
    show_progress=True
).squeeze(0)

target = 1
attrs_1 = lr_lime.attribute(
    torch.from_numpy(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)][y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)]==target]).to(torch.float32),
    target=target,
    n_samples=5000,
    perturbations_per_eval=200,
    show_progress=True
).squeeze(0)

np.save("./XAI_value/lime_tdvscp_weight_ResNet_0", attrs_0)
np.save("./XAI_value/lime_tdvscp_weight_ResNet_1", attrs_1)


# Evaluate the class 0 (TD) and class 1 (CPu) in the test set by Integrated Gradients and save

target=0
ig = IntegratedGradients(classifer.model)
inputs = torch.from_numpy(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)][y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)]==target]).to(torch.float32)
attrs_0 = ig.attribute(inputs,target=target)

target=1
ig = IntegratedGradients(classifer.model)
inputs = torch.from_numpy(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)][y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)]==target]).to(torch.float32)
attrs_1 = ig.attribute(inputs,target=target)

np.save("./XAI_value/ig_tdvscp_weight_ResNet_0", attrs_0)
np.save("./XAI_value/ig_tdvscp_weight_ResNet_1", attrs_1)

# Evaluate the class 0 (TD) and class 1 (CPu) in the test set by DeepShap and save

target=0

inputs = torch.from_numpy(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)][y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)]==target]).to(torch.float32)

inputs = torch.split(inputs, split_size_or_sections=math.ceil(len(inputs)/20), dim=0)


attrs_0_all = []

for i in inputs:
    
    dls = DeepLiftShap(classifer.model)
    i.requires_grad=True
    attrs_0 = dls.attribute(i,target=target,baselines=i*0)
    attrs_0_all.append(attrs_0.detach().numpy())

target=1

inputs = torch.from_numpy(X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)][y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)]==target]).to(torch.float32)

inputs = torch.split(inputs, split_size_or_sections=math.ceil(len(inputs)/20), dim=0)


attrs_1_all = []

for i in inputs:
    
    dls = DeepLiftShap(classifer.model)
    i.requires_grad=True
    attrs_1 = dls.attribute(i,target=target,baselines=i*0)
    attrs_1_all.append(attrs_1.detach().numpy())

np.save("./XAI_value/DeepLiftShap_tdvscp_weight_resnet_0", np.concatenate(attrs_0_all))
np.save("./XAI_value/DeepLiftShap_tdvscp_weight_resnet_1", np.concatenate(attrs_1_all))
