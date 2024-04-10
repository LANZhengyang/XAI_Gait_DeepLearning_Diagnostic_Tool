# Evaluation of forward/ backward Feature selection
# Full selection: The ranking of best angles and the accuracy achieved will be 
# First round selection: The ranking of the angles and the accuracy achieved for each angle during the first round's selection will be saved


import numpy as np
import os

from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
# Load dataset
dataset = "archives/AQM/TDvsPC_new_no_shuffle/"
d_file_list = ['CP.npz','TD.npz']
x_list, nb_classes = load_dataset_v1(dataset, d_file_list, channel_first=True,flatten=False)

# Load subject cycle table and split the dataset to training, validation and test set
idx_file_list = ['DataCP - By Subject.csv','DataTD - By Subject.csv']
x_train, x_test, y_train, y_test, x_val, y_val, cycle_end_idx, X_d, y_d = generate_data_for_train(x_list, idx_file_list,order=[1,0])

# Create folder to save the results
os.makedirs('XAI_ranking', exist_ok=True)
os.makedirs('FS_result', exist_ok=True)


# # Evaluation of forward Feature selection

for Net_name in ['ResNet','LSTM','InceptionTime']:
    print('------------------------'+Net_name+'------------------------')
    # Read the best angles list
    n_list_int = np.load('./XAI_ranking/FS_forward_'+str(Net_name)+'_TDvsCPu.npy')
    ac_list_train = []
    # Evaluation of forward Feature selection for all best angles
    for idx, i in enumerate(range(len(n_list_int))):


        i_best = n_list_int[idx]
        best_list = n_list_int[:idx]

        list_angle = list(range(22))

        best_list_str = ''
        for i in best_list:
            best_list_str+=str(i)+"_"

        list_i_m_accuray_nets = []
        i_angle = i_best

        D_name = "TDvsCPu_full_angle_"+best_list_str+str(i_angle)

        list_accuracy_nets = []

        for i_net_name in [Net_name]:
            Net_name = i_net_name
            out = np.load( './model/FS_forward/new_train_lrR_ET_'+D_name+'_'+Net_name+'/prediction.npy')
            results_cycles = print_result_bundle_i_cycles(out, y_d, x_test, cycle_end_idx,label_list=['TD', 'CPu'])
            list_accuracy_nets.append(np.array(results_cycles)[:,0])


        ac_list_train.append(list_accuracy_nets[0][0])

    os.makedirs('FS_result', exist_ok=True)
    np.save('./FS_result/FS_forward_accuracy_'+str(Net_name),ac_list_train)
    print(Net_name+'_accuracy', ac_list_train)
    # Evaluation of forward Feature selection for first round selection
    list_i_m_accuray_nets = []
    for i_angle in range(22):

        D_name = "TDvsCPu_full_angle_"+str(i_angle)

        list_accuracy_nets = []

        for i_net_name in [Net_name]:
            Net_name = i_net_name
            out = np.load( './model/FS_forward/new_train_lrR_ET_'+"TDvsCPu_full_angle_"+str(i_angle)+'_'+Net_name+'/prediction.npy')
            results_cycles = print_result_bundle_i_cycles(out, y_d, x_test, cycle_end_idx,label_list=['TD', 'CPu'])
            list_accuracy_nets.append(np.array(results_cycles)[:,0])

        list_i_m_accuray_nets.append(np.mean(list_accuracy_nets,axis=0)[0])

    best_angle = np.argmax(list_i_m_accuray_nets)

    np.save('./FS_result/FS_forward_accuracy_first_'+str(Net_name),list_i_m_accuray_nets)
    np.save('./XAI_ranking/FS_forward_ranking_first_'+str(Net_name), np.argsort(-np.array(list_i_m_accuray_nets)))
    print(Net_name+'_accuracy_first_round', list_i_m_accuray_nets)


# # Evaluation of backward Feature selection

for Net_name in ['ResNet','LSTM','InceptionTime']:
    print('------------------------'+Net_name+'------------------------')
    # Load the best angles list
    n_list_int = np.load('./XAI_ranking/FS_backward_'+str(Net_name)+'_TDvsCPu.npy')
    ac_list_train = []
    # Evaluation of backward Feature selection for all best angles
    for idx, i in enumerate(range(len(n_list_int)-1)):


        i_best = n_list_int[idx]
        best_list = n_list_int[:idx]

        list_angle = list(range(22))

        best_list_str = ''
        for i in best_list:
            best_list_str+=str(i)+"_"

        list_i_m_accuray_nets = []
        i_angle = i_best

        D_name = "TDvsCPu_full_angle_"+best_list_str+str(i_angle)

        list_accuracy_nets = []

        for i_net_name in [Net_name]:
            Net_name = i_net_name
            out = np.load( './model/FS_backward/new_train_lrR_ET_'+D_name+'_'+Net_name+'/prediction.npy')
            results_cycles = print_result_bundle_i_cycles(out, y_d, x_test, cycle_end_idx,label_list=['TD', 'CPu'])
            list_accuracy_nets.append(np.array(results_cycles)[:,0])


        ac_list_train.append(list_accuracy_nets[0][0])

    os.makedirs('FS_result', exist_ok=True)
    np.save('./FS_result/FS_backward_accuracy_'+str(Net_name),ac_list_train)
    print(Net_name+'_accuracy', ac_list_train)
    
    # Evaluation of backward Feature selection for first round selection
    list_i_m_accuray_nets = []
    for i_angle in range(22):

        D_name = "TDvsCPu_full_angle_"+str(i_angle)

        list_accuracy_nets = []

        for i_net_name in [Net_name]:
            Net_name = i_net_name
            out = np.load( './model/FS_backward/new_train_lrR_ET_'+"TDvsCPu_full_angle_"+str(i_angle)+'_'+Net_name+'/prediction.npy')
            results_cycles = print_result_bundle_i_cycles(out, y_d, x_test, cycle_end_idx,label_list=['TD', 'CPu'])
            list_accuracy_nets.append(np.array(results_cycles)[:,0])

        list_i_m_accuray_nets.append(np.mean(list_accuracy_nets,axis=0)[0])

    best_angle = np.argmax(list_i_m_accuray_nets)

    np.save('./FS_result/FS_backward_accuracy_first_'+str(Net_name),list_i_m_accuray_nets)
    np.save('./XAI_ranking/FS_backward_ranking_first_'+str(Net_name), np.argsort(np.array(list_i_m_accuray_nets)))
    print(Net_name+'_accuracy_first_round', list_i_m_accuray_nets)

