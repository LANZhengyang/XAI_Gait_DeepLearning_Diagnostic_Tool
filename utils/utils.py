import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from classifiers import ResNet
from classifiers import LSTM
from classifiers import InceptionTime
from sklearn.model_selection import train_test_split



def init_model(model_name=None,model=None,nb_classes=2 ,lr =0.001, lr_factor = 0.5, lr_patience=50,lr_reduce=True, batch_size=64, earlystopping=False, et_patience=10, max_epochs=50, gpu = [0], default_root_dir=None):
    '''
    Initialize the model and choose which kind model to use.
    It can be the trained model. no-trained model, path of trained model
    The details of the model can be set here

    Parameter:
    model_name: Type of network to load 
    model: the saved model or None to create/load model
    n_classes: number of classes
    lr: learning rate
    lr_factor: learning rate factor when learning rate reduce applied
    lr_reduce: True or False. Apply the learning rate recude or not.
    lr_patience: the learning rate will be reduce after the number of patience of epoch
    batch_size: batch size for the traning of networks
    earlystopping: True or False. Use the earlystopping or not. 
    et_patience: the patience for earlystopping to stop training
    max_epochs: max epochs for training
    gpu: index of GPU to use.
    default_root_dir: the dir path of model to be loaded

    Return: the model chosen
    '''
    if model_name == None:
        return model
    elif model_name != None:
        # Create new ResNet model
        if model_name == 'ResNet':
            return ResNet.Classifier_ResNet(nb_classes=nb_classes, lr=lr, lr_factor=lr_factor, lr_patience=lr_patience,lr_reduce=lr_reduce, batch_size=batch_size, earlystopping=earlystopping, et_patience= et_patience, max_epochs=max_epochs, gpu = gpu, default_root_dir=default_root_dir)
        # Create new LSTM model
        elif model_name == 'LSTM':
            return LSTM.Classifier_LSTM(nb_classes=nb_classes, lr=lr, lr_factor=lr_factor, lr_patience=lr_patience,lr_reduce=lr_reduce, batch_size=batch_size, earlystopping=earlystopping, et_patience= et_patience, max_epochs=max_epochs, gpu = gpu, default_root_dir=default_root_dir)
        # Create new InceptionTime model
        elif model_name == 'InceptionTime':
            return InceptionTime.Classifier_InceptionTime(nb_classes=nb_classes, lr=lr, lr_factor=lr_factor, lr_patience=lr_patience,lr_reduce=lr_reduce, batch_size=batch_size, earlystopping=earlystopping, et_patience= et_patience, max_epochs=max_epochs, gpu = gpu, default_root_dir=default_root_dir)

def find_cycles_idx_by_patient_idx(patient_idx,cycle_end_idx):
    '''
    Find the cycles index from a given patient index

    Parameter:
    patient_idx: patient index 
    cycle_end_idx: list of accumulated index of last cycle for patients

    Return: cycles index
    '''
    if patient_idx == 0:
        return np.arange(0,cycle_end_idx[patient_idx])
    else:
        return np.arange(cycle_end_idx[patient_idx-1],cycle_end_idx[patient_idx])


def patients_idx_to_cycles_idx(patients_idx,cycle_end_idx):
    '''
    Find the cycles index from given patients index

    Parameter:
    patients_idx: list of patient index 
    cycle_end_idx: list of accumulated index of last cycle for patients

    Return: cycles index
    '''
    cycles_idx = []
    for i in patients_idx:
        cycles_idx.append(find_cycles_idx_by_patient_idx(i,cycle_end_idx))
        
    return np.concatenate(cycles_idx)


def find_patients_idx_by_cycles_idx(cycles_idx,cycle_end_idx):
    '''
    Find the patients index from given cycles index

    Parameter:
    cycles_idx: list of cycle index 
    cycle_end_idx: list of accumulated index of last cycle for patients

    Return: patients index
    '''
    patients_idx = []
    for i in cycles_idx:
        patients_idx.append(cycle_idx_to_patient_idx(i,cycle_end_idx))
    return patients_idx


def cycle_idx_to_patient_idx(cycle_idx,cycle_end_idx):
    '''
    Find a cycle index is belong to which patient index 

    Parameter:
    cycles_idx: cycle index 
    cycle_end_idx: list of accumulated index of last cycle for patients

    Return: patient index
    '''
    return np.where(cycle_idx>cycle_end_idx)[0][-1]+1


def error_patient_cycle(x_test,y_test,y_d,cycle_end_idx,pre_list,nb_classes=2):
    '''
    Number of patients correctly classified using majority voting (Not used in this project)

    Parameter:
    x_test: index of patients for test set
    y_test: class of patients for test set
    y_d: class of all cycles
    cycle_end_idx: list of accumulated index of last cycle for patients
    pre_list: prediction of cycles for test set
    nb_classes: number of class

    Return: 
    error_order_idx: Bad predicted patients index
    n_patient_well_pre: Well predicted patients index
    '''    
    error_order_idx = [[],[]]
    error_cycles_idx = []

    y_test = y_test.astype(int)

    cum_list = np.cumsum([len(find_cycles_idx_by_patient_idx(i,cycle_end_idx)) for i in x_test])
    
    
    n_subject = np.zeros(nb_classes,dtype=int)
    
    
    n_cycle = 0
    n_patient_well_pre = [0,0]
    
    for i in range(len(x_test)):
        
        if isinstance(pre_list[find_cycles_idx_by_patient_idx(i,cum_list)],np.ndarray):
            y_pre = pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].astype(int)
        else:
            y_pre = pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].numpy().astype(int)


        y_true = y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)][find_cycles_idx_by_patient_idx(i,cum_list)].astype(int)
        
        print('-----------------------------')
        print('Subject No.',i)
        print('cycle_n=',len(find_cycles_idx_by_patient_idx(i,(cum_list))))
        compre_list,compre_list_count = np.unique(y_true ==y_pre,return_counts=True)
        
        n_error = 0
        if False in compre_list:
            n_error = compre_list_count[np.where(compre_list==False)[0][0]]
            
        print('n_error=',n_error)
        print()
        
        n_subject_one = np.zeros(nb_classes,dtype=int)
        
        
        if n_error:
            n_subject[int(y_test[i])]+=1
            n_subject_one[int(y_test[i])] += n_error

            n_cycle += n_error
        
        print('error in label '+str(y_test[i])+'=', n_subject_one[int(y_test[i])])    
        print('pre_list',y_pre)

        print('y_test[i]',y_test[i])
        print('majority:', np.argmax(np.bincount(y_pre)))
        
        if y_test[i] == np.argmax(np.bincount(y_pre)) and n_error!=0:
            n_patient_well_pre[y_test[i]]+=1
            
            print('well predict for class '+str(y_test[i]))
            
        elif n_error==0:
            n_patient_well_pre[y_test[i]]+=1
            
            print('well predict for no error')
        else:
            print('majority error for label No.',i)
            error_order_idx[y_test[i]].append(i)
            error_cycles_idx.append(np.where(y_pre!=y_test[i]))        

    print('- Origin -')   
    for i in range(nb_classes):
        print('n_subject_'+str(i)+'=',len(y_test[y_test==i]))
    print('n_subject:',len(y_test))
    print('n_patient_well_pre_majority:',n_patient_well_pre)
    print('n_cycles:',len(patients_idx_to_cycles_idx(x_test,cycle_end_idx)))

    print('- Error -')
    for i in range(nb_classes):
        print('n_subject_'+str(i)+'=',n_subject[i])

    print('n_subject:',np.sum(n_subject))
    print('n_cycles:',n_cycle)
    print('error patient idx:', error_order_idx)
    print('error_cycles_idx:', error_cycles_idx)
    return error_order_idx, n_patient_well_pre



def prediction_subject(x_test,y_test,y_d,cycle_end_idx,pre_list):
    '''
    The prediction based on the subject

    Parameter:
    x_test: index of patients for test set
    y_test: class of patients for test set
    y_d: class of all cycles
    cycle_end_idx: list of accumulated index of last cycle for patients
    pre_list: prediction of cycles for test set

    Return prediction results by patients
    '''    
    y_test = y_test.astype(int)
    cum_list = np.cumsum([len(find_cycles_idx_by_patient_idx(i,cycle_end_idx)) for i in x_test])
    pre_subject_list = []

    for i in range(len(x_test)):

        if isinstance(pre_list[find_cycles_idx_by_patient_idx(i,cum_list)],np.ndarray):
            y_pre = pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].astype(int)
        else:
            y_pre = pre_list[find_cycles_idx_by_patient_idx(i,cum_list)].numpy().astype(int)


        pre_subject_list.append(np.argmax(np.bincount(y_pre)))

        
    return pre_subject_list
    

def predict_new_cycles(input_data, D_name,Net_name,model_idx=0):
    '''
    The cycles prediction for any input cycles by selected model

    Parameter:
    input_data: the cycles needed to be predicted
    D_name:Dataset name
    Net_name: network name 
    model_idx: the model number selected to do the prediction

    Return all evaluation score computed
    '''
    filePath = "./model/new_train_lrR_ET_"+D_name+"_"+Net_name+"/"
    model_list = os.listdir(filePath)
    for idx_tmp,i in enumerate(model_list):
        if not i.startswith('model'):
            model_list.remove(i)
    index = model_idx
    print(index)
    proba_sum = np.zeros([len(input_data),2])
    for j in range(5):
        proba_sum +=batch_predict(D_name,Net_name, input_data,"./model/new_train_lrR_ET_"+D_name+"_"+Net_name+"/"+model_list[index*5+j])/5
    
    return proba_sum



def print_result_bundle_i_cycles(out,y_d,x_test,cycle_end_idx,label_list,n_model=10,binary=True):
    '''
    The evaluation of the performance based the cycles prediction

    Parameter:
    out: output of model(confidence of each class)
    y_d: class of all cycles
    x_test: index of patients for test set
    y_test: class of patients for test set
    cycle_end_idx: list of accumulated index of last cycle for patients
    label_list: list of class name
    n_model: number of model to evaluate
    binary: True of binary classification. Only accuracy and confusion matrix will be computed setting False

    Return all evaluation score computed
    '''
    ac_list = []
    f1_list = []
    recall_list = []

    recall_list = []

    specificity_list = []

    auc_list = []
    c_matrix_list = [] 

    for i in range(n_model):
        pre_lab_test = [ np.argmax(i) for i in out[i]]
        ac_list.append(accuracy_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))
        c_matrix_list.append(confusion_matrix(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))
        
        if binary==True:

            f1_list.append(f1_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))

            recall_list.append(recall_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))

            specificity_list.append(recall_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test, pos_label=0))

            auc_list.append(roc_auc_score(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],pre_lab_test))




    print("Accuracy:")
    print("Accuracy_list:", ac_list)
    print("Accuracy_mean:", np.mean(ac_list))
    print("Accuracy_std:", np.std(ac_list))
    print("---")
    
    print("Confusion matrix")
    print("Confusion matrix mean:", np.mean(c_matrix_list, axis=0))
    print("Confusion matrix std:", np.std(c_matrix_list, axis=0))
    print("---")
    
    if binary==True:

        print("F1")
        print("F1_list:",(f1_list))
        print("F1_mean:",np.mean(f1_list))
        print("F1_std:",np.std(f1_list))
        print("---")

        print("Sensitivity")
        print("Sensitivity list:", recall_list)
        print("Sensitivity mean:",np.mean(recall_list))
        print("Sensitivity std:",np.std(recall_list))
        print("---")

        print("Specificity")
        print("Specificity list:",specificity_list)
        print("Specificity mean:",np.mean(specificity_list))
        print("Specificity std:",np.std(specificity_list))
        print("---")


        print("AUC")
        print("AUC list:", auc_list)
        print("AUC mean:",np.mean(auc_list))
        print("AUC std:",np.std(auc_list))
   
    if binary==True:
    
        return [[np.mean(ac_list),np.std(ac_list)], [np.mean(recall_list), np.std(recall_list)], [np.mean(specificity_list), np.std(specificity_list)], [np.mean(f1_list),np.std(f1_list)],[np.mean(auc_list),np.std(auc_list)]]
    else: 
        return [[np.mean(ac_list),np.std(ac_list)], [np.mean(f1_list),np.std(f1_list)]]


def print_result_bundle_i_subjects(out,y_d,x_test, y_test,cycle_end_idx,label_list ,n_model=10, binary=True):
    '''
    The evaluation of the performance based the subject prediction

    Parameter:
    out: output of model(confidence of each class)
    y_d: class of all cycles
    x_test: index of patients for test set
    y_test: class of patients for test set
    cycle_end_idx: list of accumulated index of last cycle for patients
    label_list: list of class name
    n_model: number of model to evaluate
    binary: True of binary classification. Only accuracy and confusion matrix will be computed setting False

    Return all evaluation score computed
    '''
    pre_subject_list = []
    for i in range(n_model):
        pre_subject = prediction_subject(x_test,y_test,y_d,cycle_end_idx,np.array([ np.argmax(j) for j in out[i]]))
        pre_subject_list.append(pre_subject)
        com = pre_subject == y_test
        print("Error subject index of model "+str(i)+" :",np.where(com==False))
        
    
    ac_list = []
    f1_list = []
    recall_list = []

    recall_list = []

    specificity_list = []

    auc_list = []
    c_matrix_list = [] 

    for i in range(n_model):
        pre_lab_test = pre_subject_list[i]
        ac_list.append(accuracy_score(y_test,pre_lab_test))
        c_matrix_list.append(confusion_matrix(y_test,pre_lab_test))
        
        if binary==True:
            f1_list.append(f1_score(y_test,pre_lab_test))

            recall_list.append(recall_score(y_test,pre_lab_test))

            specificity_list.append(recall_score(y_test,pre_lab_test, pos_label=0))

            auc_list.append(roc_auc_score(y_test,pre_lab_test))
 
    print("Accuracy:")
    print("Accuracy_list:",ac_list)
    print("Accuracy_mean:",np.mean(ac_list))
    print("Accuracy_std:",np.std(ac_list))
    print("---")
    
    print("Confusion matrix")
    print("Confusion matrix mean:",np.mean(c_matrix_list, axis=0))
    print("Confusion matrix std:",np.std(c_matrix_list, axis=0)) 
    print("---")
    
    if binary==True:

        print("F1")
        print("F1_list:",(f1_list))
        print("F1_mean:",np.mean(f1_list))
        print("F1_std:",np.std(f1_list))
        print("---")

        print("Sensitivity")
        print("Sensitivity list:", recall_list)
        print("Sensitivity mean:",np.mean(recall_list))
        print("Sensitivity std:",np.std(recall_list))
        print("---")

        print("Specificity")
        print("Specificity list:",specificity_list)
        print("Specificity mean:",np.mean(specificity_list))
        print("Specificity std:",np.std(specificity_list))
        print("---")


        print("AUC")
        print("AUC list:", auc_list)
        print("AUC mean:",np.mean(auc_list))
        print("AUC std:",np.std(auc_list))
    
    if binary==True:
    
        return [[np.mean(ac_list),np.std(ac_list)], [np.mean(recall_list), np.std(recall_list)], [np.mean(specificity_list), np.std(specificity_list)], [np.mean(f1_list),np.std(f1_list)],[np.mean(auc_list),np.std(auc_list)]]
    else: 
        return [[np.mean(ac_list),np.std(ac_list)], [np.mean(f1_list),np.std(f1_list)]]


def merge_mean_std(mean_std):
    '''
    Merge the mean and std together and convert to str (keep the firs two digits afet the decimal point)

    Parameter:
    mean_std: mean and std list from output

    Return all evaluation score computed
    '''    
    return str('%.2f'%mean_std[0])+"±"+str('%.2f'%mean_std[1])


def output_csv(output_list):
    '''
    Merge the mean and std together and convert to str (keep the firs two digits afet the decimal point)

    Parameter:
    mean_std: mean and std list from output

    Return all evaluation score computed
    '''
    list_str = []
    for i in output_list:
        list_str.append(merge_mean_std(i))
    return list_str


def batch_predict(D_name,Net_name, input_data,model_path,nb_classes=2):
    '''
    Evaluation of one pytorch model trained by input data

    Parameter:
    D_name:Dataset name
    Net_name: network name 
    input_data: input data to feed to the model
    model_path: path of model to be evaluated
    nb_classes: number of different class

    Return prediction of model by confidence of each class
    '''
    pred_set = Dataset_torch(input_data,with_label=False)
    data_loader_pred = torch.utils.data.DataLoader(dataset=pred_set, batch_size=64,num_workers=4)
    
    
    classifer = init_model(model_name=Net_name,nb_classes=nb_classes).load(model_path, nb_classes=nb_classes)
    
    trainer = pl.Trainer(gpus=[0])
    pred = trainer.predict(model=classifer.model,dataloaders = data_loader_pred)
    pred = torch.cat(pred)
    return pred.detach().cpu().numpy()


class Dataset_torch(Dataset):
    '''
    Adapt the dataset to feed to pytorch dataloader

    Parameter:
    data: input data to adapt
    with_label: True or False, with label or not

    Return prediction of model by confidence of each class
    '''
    def __init__(self, data,with_label=True):
        self.with_label =  with_label
        
        if self.with_label:
            self.data_x, self.data_y = data
        else:
            self.data_x = data
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if self.with_label:
            return self.data_x[idx], self.data_y[idx]
        else:
            return self.data_x[idx]


        
def eval_model_5_bundle(D_name,Net_name,X_d,y_d,x_test,y_test,cycle_end_idx, number_model=10,idx=None,default_root_dir=None):
    '''
    Evaluate the model by 5 model's average's confidence

    Parameter:
    D_name:Dataset name
    Net_name: network name 
    X_d: all cycles
    y_d: all cylce's class
    x_test: patient index of test set
    y_test: patient class of test set
    cycle_end_idx: list of accumulated index of last cycle for patients
    number_model: number of total model to evaluate
    idx: None or index of model. None means evaluate all model, or evaluate the model with index provided.
    default_root_dir: path of the model

    Return prediction of model by confidence of each class
    ''' 
    if default_root_dir == None:
        default_root_dir="./model/new_train_lrR_ET_"+D_name+"_"+Net_name+"/"
    else:
        default_root_dir = default_root_dir+"new_train_lrR_ET_"+D_name+"_"+Net_name+"/"
    
    
    list_f1 = []
    list_accuracy = []    
    filePath = default_root_dir
    model_list = os.listdir(filePath)
    for idx_tmp,i in enumerate(model_list):
        if not i.startswith('model'):
            model_list.remove(i)
    print("len(model_list)",len(model_list))
    proba_sum_list = []
    if idx==None:
        for index in range(number_model):
            print(index)
            proba_sum = np.zeros([len(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)]),2])
            for j in range(5):

                print(model_list[index])

                proba_sum +=batch_predict(D_name,Net_name, X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],filePath+model_list[index*5+j])/5
            proba_sum_list.append(proba_sum)
    else:
            index = idx
            print(index)
            proba_sum = np.zeros([len(y_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)]),2])
            for j in range(5):

                print(model_list[index])

                proba_sum +=batch_predict(D_name,Net_name, X_d[patients_idx_to_cycles_idx(x_test,cycle_end_idx)],filePath+model_list[index*5+j])/5
            proba_sum_list = (proba_sum)
    return proba_sum_list


def train_model(D_name,Net_name,X_d,y_d,x_train,x_val,y_train,y_val,cycle_end_idx,nb_classes=2, lr_reduce=True, lr_factor = 0.5, earlystopping=True, lr_patience=1, batch_size=64,lr=0.001, max_epochs=50,gpu=[0],default_root_dir=None):
    '''
    Training the model

    Parameter:
    D_name:Dataset name
    Net_name: network name 
    X_d: all cycles
    y_d: all cylce's class
    x_train: patient index of traning set
    x_val: patient index of validation set
    y_train: patient class of traning set
    y_val: patient class of validation set
    cycle_end_idx: list of accumulated index of last cycle for patients
    nb_classes: number of different class
    lr_reduce: True or False. Apply the learning rate recude or not.
    lr_factor: learning rate factor when learning rate reduce applied    
    earlystopping: True or False. Use the earlystopping or not.    lr: learning rate
    lr_patience: the learning rate will be reduce after the number of patience of epoch
    batch_size: batch size for the traning of networks
    max_epochs: max epochs for training
    gpu: index of GPU to use.
    et_patience: the patience for earlystopping to stop training
    default_root_dir: path where to save the model
    

    Return None - model is saved
    '''    
    if default_root_dir == None:
        default_root_dir="./model/new_train_lrR_ET_"+D_name+"_"+Net_name+"/"
    else:
        default_root_dir = default_root_dir+"new_train_lrR_ET_"+D_name+"_"+Net_name+"/"
    a_list = []
    for i in range(5):
        print("train number -", i)
        classifer = init_model(model_name=Net_name,lr_reduce=lr_reduce, lr_factor = lr_factor, earlystopping=earlystopping, lr_patience=lr_patience, batch_size=batch_size,lr=lr, max_epochs=max_epochs,gpu=gpu,default_root_dir=default_root_dir,nb_classes=nb_classes)
        classifer.fit(X_d[patients_idx_to_cycles_idx(x_train,cycle_end_idx)], y_d[patients_idx_to_cycles_idx(x_train,cycle_end_idx)], X_d[patients_idx_to_cycles_idx(x_val,cycle_end_idx)], y_d[patients_idx_to_cycles_idx(x_val,cycle_end_idx)],ckpt_monitor='val_accuracy')
        print("train number end -", i)

      
def load_dataset_v1(dir_dataset,d_file_list,channel_first=True,flatten=True):
    '''
    Load the dataset from files

    Parameter:

    dir_dataset: path of dataset
    d_file_list: list of name of dataset's files
    channel_first: True, gait angles before times series. False, times series before gait angles
    flatten: True of False, flatten the data

    Return 
    x_list: list of all cycles
    nb_classes: number of different class
    '''  

    index = [0,1,2,3,4,5,6,7,8,9,14,15,16,17,18,19,20,21,22,23,24,29]
    
    x_list = []
    
    for i in d_file_list:
    
        npzfile = np.load(dir_dataset + i)

        x_all = npzfile['Input']

        x = x_all[:,:,index]
        if channel_first:
            x = x.transpose(0,2,1)
            
        if flatten == True:
            x = x.reshape([x.shape[0],-1])
        
        x_list.append(x)

    nb_classes = len(d_file_list)
    
    return x_list, nb_classes



def generate_data_for_train(x_list, idx_file_list,order,random_state=[0,0]):
    '''
    Generate training set, validation set and test set

    Parameter:
    x_list: list of all cycles
    idx_file_list: list of the csv file's name saved the list of accumulated index of last cycle for patients for each class(in second column) in order
    order: order of the class loaded
    random_state: random seed list for data split

    Return 
    x_train: patient index in traning set 
    x_test: patient index in test set 
    y_train: patient class in training set
    y_test: patient class in test set
    x_val: patient index in validation set 
    y_val: patient class in validation set
    cycle_end_idx: list of accumulated index of last cycle for patients
    X_d: all cycles
    y_d: all cycle's class
    '''

    X_d = np.concatenate(x_list)
    y_d = []
    idx_list = []
    Flag = True
    
    for idx,i in enumerate(idx_file_list):
        
        if Flag == True:
            idx_0 = pd.read_csv(i, header=None)[1].to_numpy()
            idx_list.append(idx_0)
            y_d.append(np.zeros((idx_0[-1]))+order[idx])
            
        else:
            idx_new = pd.read_csv(i, header=None)[1].to_numpy()
            idx_list.append(idx_new+idx_list[-1][-1])
            y_d.append(np.zeros((idx_new[-1]))+order[idx])

        Flag = False
            
            
    cycle_end_idx = np.concatenate(idx_list)
 
    patient_index_range = np.arange(len(cycle_end_idx))    
    
    patient_class_list = []
    for idx,i in enumerate(idx_list):
        
        patient_class_list.append(np.zeros(len(i))+order[idx])
    
    patient_class = np.concatenate(patient_class_list)
    y_d = np.concatenate(y_d)    
    
    x_train, x_test, y_train, y_test = train_test_split(patient_index_range, patient_class, test_size=0.4, stratify=patient_class,random_state=random_state[0])
    x_val, x_test, y_val, y_test = train_test_split(x_test,y_test, test_size=0.75, random_state=random_state[1], stratify=y_test)
       
    return x_train, x_test, y_train, y_test, x_val, y_val, cycle_end_idx, X_d, np.array(y_d)


def plot_cycle_with_idx(n_subject, name_dataset, RL_list, cycle_end_idx, X_d):
    '''
    Plot the cycles by given index

    Parameter:
    n_subject: patient index to plot
    name_dataset: name of dataset
    RL_list: list of description of each side
    cycle_end_idx: list of accumulated index of last cycle for patients
    X_d: All cycles from dataset

    Return None, the figure plot directly
    '''
    cycles_idx = find_cycles_idx_by_patient_idx(n_subject,cycle_end_idx)
    X_d_all = X_d[cycles_idx]
    d_mean_all = np.mean(X_d_all,0)

    fig, axs = plt.subplots(6, 7, figsize=(70, 40))

    title_list = ['F/E','Ab/Ad','Rot I/E']

    class_list = [name_dataset+' - subject No.'+str(n_subject)]



    for i,ax in enumerate(axs.flatten(), start=0):
        if i<6:
            axs.flatten()[i].axis('off')
            axs.flatten()[i].text(0.15,0.1,title_list[i%3],size=60,weight='bold')
            if i%3==1:
                axs.flatten()[i].text(0.0,0.6,RL_list[(i-1)//3],size=60,weight='bold')

    list_right = ['','Pelvis','Hip','Knee','Ankle','Foot']
    for idx,i in enumerate([6,13,20,27,34,41]):  
        axs.flatten()[i].axis('off')
        axs.flatten()[i].text(0.1,0.5,list_right[idx],size=60,weight='bold')

    for i in range(7):
        ht_line = plt.axes([0.065, 0.9-i*0.133, 0.89,0.01])
        ht_line.plot([0,1],[0.5,0.5],linewidth = 10,c='black')
        ht_line.axis('off')
        if i == 0:
            ht_line.text( 2.7/7,0.3,class_list[0],size=60,weight='bold')

    for i in range(8):
        linewidth = 5
        high_line = 0.73

        if i in [0,6,7]:
            linewidth_use = 10
            high_line_use = 0.87
        elif i in [3,9]:
            linewidth_use = 10
            high_line_use = 0.8
        else:
            linewidth_use = linewidth
            high_line_use = high_line

        t_line = plt.axes([0.1+i*0.116, 0.07, 0.01, high_line_use])
        t_line.plot([0.5,0.5],[0,1],linewidth = linewidth_use,c='black')
        t_line.axis('off')


    plt.subplots_adjust(wspace=0.5, 
                        hspace=0.5)


    for idx,i in enumerate([7,8,9,14,15,16,21,22,23,28,37 , 7+3,8+3,9+3,14+3,15+3,16+3,21+3,22+3,23+3,28+3,37+3 ]):

        if i in [7,14,21,28, 7+3,14+3,21+3,28+3]:
            axs.flatten()[i].set_ylim(-20,70)
            axs.flatten()[i].set_yticks([-20,0,20,40,70], size = 50)
        else:
            axs.flatten()[i].set_ylim(-45,45)
            axs.flatten()[i].set_yticks([-45,-25,-10,0,10,25,45], size = 50)
        for idx_jj,jj in enumerate(X_d_all):
            if idx_jj == 0:
                label = 'Cycles'
            else:
                label = None

            tmp = axs.flatten()[i].plot(range(101),jj[idx], label=label,c='b',linewidth=1)


        tmp = axs.flatten()[i].plot(range(101),d_mean_all[idx],label='Avg',c='r',linewidth=3)
        plt.subplots_adjust(left=None, bottom=None, right=0.91, top=0.9,hspace=0.45,wspace=0.3)
        axs.flatten()[i].spines['bottom'].set_linewidth(3)
        axs.flatten()[i].spines['left'].set_linewidth(3)
        axs.flatten()[i].spines['right'].set_linewidth(3)
        axs.flatten()[i].spines['top'].set_linewidth(3)

        axs.flatten()[i].set_ylabel('Angle',weight= 'bold',size= 20)
        axs.flatten()[i].set_xlabel('% gait cycle',weight= 'bold',size= 20)
        axs.flatten()[i].legend(prop = {'size':20})

        axs.flatten()[i].tick_params(labelsize=20)

    for i in np.arange(1,7)*7-1:
        axs.flatten()[i].axis('off')

    for i in [29,30,35,36]:
        for j in range(2):
            axs.flatten()[i+j*3].axis('off')
            
def XAI_plot(attrs_0_norm,attrs_1_norm,RL_list, D_name,Net_name,labels,XAI_method,path_img = './XAI_plot/'):
    '''
    Plot the XAI results and save the figure

    Parameter:
    attrs_0_norm: normalized relevance value for class 0
    attrs_1_norm: normalized relevance value for class 1
    RL_list: RL_list: list of description of each side
    D_name: name of dataset
    Net_name: name of network 
    labels: list of name of class
    XAI_method: name of XAI method
    path_img: path to save the result

    Return None, the figure plot and save directly
    '''

    fig, axs = plt.subplots(6, 7, figsize=(70, 40))

    title_list = ['F/E','Ab/Ad','Rot I/E']

    class_list = [str(D_name)+' - '+ str(XAI_method) +' - '+str(Net_name)]

    RL_list = RL_list

    for i,ax in enumerate(axs.flatten(), start=0):
        if i<6:
            axs.flatten()[i].axis('off')
            axs.flatten()[i].text(0.15,0.1,title_list[i%3],size=60)
            if i%3==1:
                axs.flatten()[i].text(0.0,0.6,RL_list[(i-1)//3],size=60)

    list_right = ['','Pelvis','Hip','Knee','Ankle','Foot']
    for idx,i in enumerate([6,13,20,27,34,41]):  
        axs.flatten()[i].axis('off')
        axs.flatten()[i].text(0.1,0.5,list_right[idx],size=60)

    for i in range(7):
        ht_line = plt.axes([0.065, 0.9-i*0.133, 0.89,0.01])
        ht_line.plot([0,1],[0.5,0.5],linewidth = 10,c='black')
        ht_line.axis('off')
        if i == 0:
            ht_line.text( 2.7/7,0.3,class_list[0],size=60)

    for i in range(8):
        linewidth = 5
        high_line = 0.73

        if i in [0,6,7]:
            linewidth_use = 10
            high_line_use = 0.87
        elif i in [3,9]:
            linewidth_use = 10
            high_line_use = 0.8
        else:
            linewidth_use = linewidth
            high_line_use = high_line

        t_line = plt.axes([0.1+i*0.116, 0.07, 0.01, high_line_use])
        t_line.plot([0.5,0.5],[0,1],linewidth = linewidth_use,c='black')
        t_line.axis('off')


    plt.subplots_adjust(wspace=0.5, 
                        hspace=0.5)



    for idx,i in enumerate([7,8,9,14,15,16,21,22,23,28,37 , 7+3,8+3,9+3,14+3,15+3,16+3,21+3,22+3,23+3,28+3,37+3 ]):



        axs.flatten()[i].set_ylim(0,2.2)

        axs.flatten()[i].plot(range(101),np.abs(attrs_0_norm)[idx],c='orange',label=labels[0])
        axs.flatten()[i].plot(range(101),np.abs(attrs_1_norm)[idx],c='blue',label=labels[1])
        axs.flatten()[i].plot(range(101),np.abs(attrs_1_norm)[idx]+np.abs(attrs_0_norm)[idx],c='green',label='Total')


        axs.flatten()[i].set_ylabel('XAI relevance value')
        axs.flatten()[i].set_xlabel('% gait cycle')
        axs.flatten()[i].legend()

    for i in np.arange(1,7)*7-1:
        axs.flatten()[i].axis('off')

    for i in [29,30,35,36]:
        for j in range(2):
            axs.flatten()[i+j*3].axis('off')
    os.makedirs(path_img, exist_ok=True)
    path_img = path_img
    save=str(XAI_method)+'_'+str(Net_name)+'_'+str(D_name)
    plt.savefig(path_img+save,bbox_inches = 'tight',facecolor ="w",dpi=100)
