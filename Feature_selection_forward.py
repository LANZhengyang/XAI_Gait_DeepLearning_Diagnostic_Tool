import numpy as np
import os
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import copy

# Load dataset
dataset = "archives/AQM/TDvsPC_new_no_shuffle/"
d_file_list = ['CP.npz','TD.npz']
x_list, nb_classes = load_dataset_v1(dataset, d_file_list, channel_first=True,flatten=False)

# Load subject cycle table and split the dataset to training, validation and test set
idx_file_list = ['DataCP - By Subject.csv','DataTD - By Subject.csv']
x_train, x_test, y_train, y_test, x_val, y_val, cycle_end_idx, X_d, y_d = generate_data_for_train(x_list, idx_file_list,order=[1,0])

# Set the path to save the model
default_root_dir = './model/FS_forward/'

# Set the basic information for training
D_name_str = 'TDvsCPu'
Net_name = 'ResNet'

# Copy the original data
X_d_original = X_d.copy()

# Best angle for each selection
i_best = -1
# List of best angle for each selection
best_list = []

# List of angle not yet be selected
list_angle = list(range(22))

for list_angle_i in range(len(list_angle)):
    
    # Add the best angle to the list and remove from to-be-selected list
    if list_angle_i!=0:
        best_list = best_list+[i_best]
        list_angle.remove(i_best)
        
    
    # generate the str of best angles
    if list_angle_i!=0:
        best_list_str = ''
        for i in best_list:
            best_list_str+=str(i)+"_"
    else:
        best_list_str =''

    # select the angles from to-be-selected list
    for i_angle in list_angle:
        # Replance the angle not yet be selected to zeros
        X_d_keep = X_d_original[:,best_list+[i_angle],:].copy()
        X_d = np.zeros_like(X_d)
        X_d[:,best_list+[i_angle],:] = X_d_keep

        # generate the name of the dataset with the angle used
        D_name = D_name_str+"_full_angle_"+best_list_str+str(i_angle)

        # train 10 models
        print('Training angle number: '+best_list_str+str(i_angle))
        for j in range(10):
            train_model(D_name=D_name,Net_name=Net_name,X_d=X_d,y_d=y_d,x_train=x_train,x_val=x_val,y_train=y_train,y_val=y_val,cycle_end_idx=cycle_end_idx,default_root_dir=default_root_dir)
        # evaluate 10 models
        out = eval_model_5_bundle(D_name=D_name,Net_name=Net_name,X_d=X_d,y_d=y_d,x_test=x_test,y_test=y_test,cycle_end_idx=cycle_end_idx,default_root_dir=default_root_dir)
        np.save('./model/FS_forward/new_train_lrR_ET_'+D_name_str+"_full_angle_"+best_list_str+str(i_angle)+'_'+Net_name+'/prediction.npy' ,out)

    # compute the mean of accuracy of 10 models and select the angle added then achieved best accuracy as best angle
    list_i_m_accuray_nets = []
    for i_angle in list_angle:

        D_name = D_name_str+"_full_angle_"+best_list_str+str(i_angle)

        list_accuracy_nets = []
        out = np.load( default_root_dir+'new_train_lrR_ET_'+D_name+'_'+Net_name+'/prediction.npy')
        results_cycles = print_result_bundle_i_cycles(out, y_d, x_test, cycle_end_idx,label_list=['TD', 'CPu'])
        list_accuracy_nets.append(np.array(results_cycles)[:,0])

        list_i_m_accuray_nets.append(np.mean(list_accuracy_nets,axis=0)[0])

    best_angle = np.argmax(list_i_m_accuray_nets)

    print(best_angle)
    i_best = np.array(list_angle)[np.argsort(-np.array(list_i_m_accuray_nets))][0]
    print(i_best)
# save best angles list 
os.makedirs('XAI_ranking', exist_ok=True)
np.save('./XAI_ranking/FS_forward_'+Net_name+'_'+D_name_str,np.array(best_list+[i_best]))
