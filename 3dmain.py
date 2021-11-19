import copy
import cv2 as cv
import gc
import io
import linecache
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sn
import time
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import warnings

from classes.efmodel import EFModel
from classes.identity import Identity
from classes.lfmodel import LFModel
from classes.violent_dataset import ViolentDataset

from collections import Counter
from datetime import datetime
from matplotlib import cm, figure
#from memory_profiler import profile
from PIL import Image
from python_speech_features import mfcc
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
from sklearn import svm
from sklearn.metrics import confusion_matrix, zero_one_loss


data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

warnings.filterwarnings("ignore")

audio_extract_timer = []
audio_extraction_timer = []

video_timer_ep = []
video_timer = []

models_list = ['resnet18','resnet34','r3d_18','mc3_18','r2plus1d_18']

types_of_files = ['nv','v']

fc_models = ['resnet18','resnet34','mc3_18','r2plus1d_18']

vision_models = ['r3d_18','mc3_18','r2plus1d_18']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters and others...

audio_model = 'resnet34'

video_model = 'mc3_18'

criterion = nn.BCEWithLogitsLoss()

learn_rate = 1e-2

num_epochs = 1

num_videos = 1

b_size = 1 # Pouca RAM

# STATS HOLDERS

best_a_acc = 0.0

best_v_acc = 0.0

best_ef_acc = 0.0

best_lf_acc = 0.0


visited = {}

# timers
epoch_timer = []
audio_extract_timer = []
audio_pr_timer = []

# audio model stats
a_train_accu = []
a_train_loss = []
a_val_accu = []
a_val_loss = []
a_true = []
a_pred = []

# video model stats
v_train_accu = []
v_train_loss = []
v_val_accu = []
v_val_loss = []
v_true = []
v_pred = []

# ef model stats
ef_train_accu = []
ef_train_loss = []
ef_val_accu = []
ef_val_loss = []

# lf model stats
lf_train_accu = []
lf_train_loss = []
lf_val_accu = []
lf_val_loss = []


def plot_data(since, num_epochs, num_videos, learn_rate, batch_size, train_accu, val_accu, train_loss, val_loss, model_name, model_type, best_acc):
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y-%Hh%Mm%Ss")
    time_elapsed = time.time() - since

    fig = figure.Figure()
    plt.plot(train_accu,'-o')
    plt.plot(val_accu,'-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Valid'])
    plt.title(f'Train vs Valid Accuracy.\n Batch_size: {batch_size}. Learning Rate: {learn_rate}. {round(time_elapsed/(num_epochs*2*num_videos),2)} secs / video.')

    plt.savefig('plots/{}/{}/{}_train_val_acc'.format(model_type,model_name,dt_string))

    fig.clear()
    plt.cla()

    plt.plot(train_loss,'-o')
    plt.plot(val_loss,'-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title(f'Train vs Valid Losses.\n Batch_size: {batch_size}. Learning Rate: {learn_rate}.')

    plt.savefig('plots/{}/{}/{}_train_val_loss'.format(model_type, model_name,dt_string))
    
    fig.clear()
    plt.cla()
    plt.close(fig)

    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}\n'.format(best_acc))

def time_plot(times_arr,name,title):
    fig = figure.Figure()
    plt.plot(times_arr,'-o')
    plt.xlabel('epoch')
    plt.ylabel('time')
    plt.legend('Time')
    plt.title(title)
    plt.savefig(name)
    fig.clear()
    plt.cla()
    plt.close(fig)

def save_model(type, epoch_acc, best_epoch_acc, save_name, model):
    print(f'{type}. Epoch: {epoch_acc} Best: {best_epoch_acc}')
    if ((epoch_acc > best_epoch_acc) or (not(os.path.isfile(save_name)))):
        best_epoch_acc = epoch_acc
        torch.save(model,save_name)

def get_models(model_name):
    # get model used from switcher
    if model_name not in vision_models:
        model = getattr(models, model_name)(pretrained=True)
    else:
        model = getattr(models.video, model_name)(pretrained=True)

    # change predictive layer from 1000 to 2 out_features
    if model_name in fc_models:
        in_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_ftrs,2)
        )
    else:
        last_idx = len(model.classifier)-1
        in_ftrs = model.classifier[last_idx].in_features
        model.classifier[last_idx] = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_ftrs,2)
        )

    return model

def prediction(optimizer, model, input1, input2, criterion, labels, phase, running_loss, running_corrects):
    optimizer.zero_grad()

    # forward + backward + optimize
    if input2 is None:
        # Audio and Video Models
        outputs = model(input1)
    else:
        # EF Model
        outputs = model(input1,input2)

    pred = torch.argmax(outputs, 1)
    loss = criterion(outputs, labels)
    
    if phase == 'train':
        loss.backward()
        optimizer.step()

    running_loss += loss.item() * input1.size(0)
    running_corrects += torch.sum(pred == torch.argmax(labels,1))

    return outputs, running_loss, running_corrects, pred

def get_lf_input(audio_output, video_output, device):
    # Average the predictions out
    stack = torch.cat((audio_output[0],video_output[0]),0) #stack the predictions
    empty = torch.empty((0,4)).to(device) # create empty tensor in order to create correct shape, in case there is only one prediction
    x = torch.vstack((stack,empty))

    for i in range(1,len(audio_output)):
        x = torch.vstack((x,torch.cat((audio_output[i],video_output[i]),0)))

    return x

def write_results(audio_model,video_model,ef_model,batch_size,learn_rate,a_t_acc,a_t_loss,a_v_acc,a_v_loss,v_t_acc,v_t_loss,v_v_acc,v_v_loss,ef_t_acc,ef_t_loss,ef_v_acc,ef_v_loss,lf_t_acc,lf_t_loss,lf_v_acc,lf_v_loss):
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime(f'{audio_model}_{video_model}_{batch_size}_{learn_rate}_%d_%m_%Y-%Hh%Mm%Ss')

    with open(f'results/{dt_string}.txt', 'x') as f:
        f.write(f'Models:\n')
        f.write(f'Audio: {audio_model}\n')
        f.write(f'Video: {video_model}\n')
        f.write(f'Early Fusion: {ef_model}\n')
        f.write(f'\nParameters:\n')
        f.write(f'Batch Size: {batch_size} & Learning Rate: {learn_rate}\n\n---\n')
        f.write(f'TRAIN:\n')
        f.write(f'\nAccuracies:\n')
        f.write(f'Audio: {a_t_acc}\n')
        f.write(f'Video: {v_t_acc}\n')
        f.write(f'Early Fusion: {ef_t_acc}\n')
        f.write(f'Late Fusion: {lf_t_acc}\n---\n')
        f.write(f'\nLosses:\n')
        f.write(f'Audio: {a_t_loss}\n')
        f.write(f'Video: {v_t_loss}\n')
        f.write(f'Early Fusion: {ef_t_loss}\n')
        f.write(f'Late Fusion: {lf_t_loss}\n\n---\n')
        f.write(f'VALIDATION:\n')
        f.write(f'\nAccuracies:\n')
        f.write(f'Audio: {a_v_acc}\n')
        f.write(f'Video: {v_v_acc}\n')
        f.write(f'Early Fusion: {ef_v_acc}\n')
        f.write(f'Late Fusion: {lf_v_acc}\n---\n')
        f.write(f'\nLosses:\n')
        f.write(f'Audio: {a_v_loss}\n')
        f.write(f'Video: {v_v_loss}\n')
        f.write(f'Early Fusion: {ef_v_loss}\n')
        f.write(f'Late Fusion: {lf_v_loss}\n---')
        f.close()

def plot_cf_matrix(cm_true, cm_pred,name):
    classes = ('nv','v')
    cf_matrix = confusion_matrix(cm_true,cm_pred)
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes], columns = [i for i in classes])
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm,annot=True,cmap='YlGnBu')
    plt.savefig(name)
    plt.cla()
    plt.clf()

def main():
    # Folder creation
    for t in ['models','plots/late_fusion/lf','results']:
        try:
    	    os.makedirs(t)
        except FileExistsError:
	    # directory already exists
            pass
    
    for t in ['audio','video']:
        for m in models_list:
            try:
                os.makedirs('plots/{}/{}'.format(t,m))
            except FileExistsError:
                # directory already exists
                pass
    
    for t in ['early_fusion']:
        for m in models_list:
            for n in models_list:
                if m != n:
                    try:
                        os.makedirs('plots/{}/{}_{}'.format(t,m,n))
                    except FileExistsError:
                        # directory already exists
                        pass
    
    # Instantiate datasets
    train_dataset = ViolentDataset('data/videos/train')
    val_dataset = ViolentDataset('data/videos/val')


    # GET MODELS
    model_a = get_models(audio_model) # Audio
    model_v = get_models(video_model) # Video

    ef_a = get_models(audio_model) # EF
    ef_a.fc = Identity()

    ef_v = get_models(video_model)
    ef_v.fc = Identity()

    model_ef = EFModel(ef_a, ef_v, device)
    model_lf = LFModel(4) # LF


    model_a = model_a.to(device)
    model_v = model_v.to(device)
    model_ef = model_ef.to(device)
    model_lf = model_lf.to(device)

    # Observe that all parameters are being optimized
    model_a_optimizer = optim.SGD(model_a.parameters(), lr=learn_rate, momentum=0.9)
    model_v_optimizer = optim.SGD(model_v.parameters(), lr=learn_rate, momentum=0.9)
    model_ef_optimizer = optim.SGD(model_ef.parameters(), lr=learn_rate, momentum=0.9)
    model_lf_optimizer = optim.SGD(model_lf.parameters(), lr=learn_rate, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    model_a_scheduler = lr_scheduler.StepLR(model_a_optimizer, step_size=2, gamma=0.1)
    model_v_scheduler = lr_scheduler.StepLR(model_v_optimizer, step_size=2, gamma=0.1)
    model_ef_scheduler = lr_scheduler.StepLR(model_ef_optimizer, step_size=2, gamma=0.1)
    model_lf_scheduler = lr_scheduler.StepLR(model_lf_optimizer, step_size=2, gamma=0.1)

    since = time.time()

    print(f'\n------\nAudio Model: {audio_model}\n------\n\n------\nVideo Model: {video_model}\n------\n')

    for epoch in range(0,num_epochs): # num epochs
        epoch_time = time.time()
        visited = {}
        dataset_sizes = {}

        a_predictions = []
        v_predictions = []
        ef_predictions = []
        lf_predictions = []
        cm_labels = []

        for phase in ['train','val']: # phase
            dataset_sizes[phase] = 0
            print(f'E: [{epoch+1}] P: [{phase}]')

            if phase == 'train':
                dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, num_workers=0)
            else:
                dataloader = DataLoader(val_dataset, batch_size=b_size, shuffle=True, num_workers=0)

            if phase == 'train':
                model_a.train()  # Set model to training mode
                model_v.train()  # Set model to training mode
                model_ef.train()  # Set model to training mode
                model_lf.train()  # Set model to training mode
            else:
                model_a.eval()   # Set model to evaluate mode
                model_v.eval()   # Set model to evaluate mode
                model_ef.eval()   # Set model to evaluate mode
                model_lf.eval()   # Set model to evaluate mode

            a_running_loss = 0.0
            a_running_corrects = 0

            v_running_loss = 0.0
            v_running_corrects = 0
            
            ef_running_loss = 0.0
            ef_running_corrects = 0

            lf_running_loss = 0.0
            lf_running_corrects = 0
        
            for idx, sample in enumerate(dataloader):
                audio, label, video = sample['audio'], sample['label'], sample['video']

                audio = audio.to(device)
                video = video.to(device)
                labels = torch.Tensor()

                for lbl in label:
                    if lbl == 'nv':
                        labels = torch.cat((labels,torch.Tensor([[1,0]])),0)
                        cm_labels.append(0)
                    else:
                        labels = torch.cat((labels,torch.Tensor([[0,1]])),0)
                        cm_labels.append(1)

                labels = labels.to(device)

                ### PREDICTIONS ###
                ### AUDIO Prediction
                audio_outputs, a_running_loss, a_running_corrects, a_pred = prediction(model_a_optimizer, model_a, audio, None, criterion, labels, phase, a_running_loss, a_running_corrects)
                a_predictions.extend(a_pred.cpu().numpy())

                ### VIDEO Prediction
                video_outputs, v_running_loss, v_running_corrects, v_pred = prediction(model_v_optimizer, model_v, video, None, criterion, labels, phase, v_running_loss, v_running_corrects)
                v_predictions.extend(v_pred.cpu().numpy())

                ### EF Prediction
                ef_outputs, ef_running_loss, ef_running_corrects, ef_pred = prediction(model_ef_optimizer, model_ef, audio, video, criterion, labels, phase, ef_running_loss, ef_running_corrects)
                ef_predictions.extend(ef_pred.cpu().numpy())

                ### LF Prediction
                lf_input = get_lf_input(audio_outputs.detach().clone().to(device),video_outputs.detach().clone().to(device),device)

                lf_outputs, lf_running_loss, lf_running_corrects, lf_pred = prediction(model_lf_optimizer, model_lf, lf_input, None, criterion, labels, phase, lf_running_loss, lf_running_corrects)
                lf_predictions.extend(lf_pred.cpu().numpy())

                dataset_sizes[phase] += audio.size(0) # Update dataset size       
            

            a_epoch_loss = a_running_loss / dataset_sizes[phase]
            a_epoch_acc = a_running_corrects.double() / dataset_sizes[phase]

            v_epoch_loss = v_running_loss / dataset_sizes[phase]
            v_epoch_acc = v_running_corrects.double() / dataset_sizes[phase]            
            
            ef_epoch_loss = ef_running_loss / dataset_sizes[phase]
            ef_epoch_acc = ef_running_corrects.double() / dataset_sizes[phase]

            lf_epoch_loss = lf_running_loss / dataset_sizes[phase]
            lf_epoch_acc = lf_running_corrects.double() / dataset_sizes[phase]

            if (phase=='train'):
                a_train_accu.append(a_epoch_acc.item())
                a_train_loss.append(a_epoch_loss)

                v_train_accu.append(v_epoch_acc.item())
                v_train_loss.append(v_epoch_loss)
                
                ef_train_accu.append(ef_epoch_acc.item())
                ef_train_loss.append(ef_epoch_loss)
                
                lf_train_accu.append(lf_epoch_acc.item())
                lf_train_loss.append(lf_epoch_loss)

            else:
                a_val_accu.append(a_epoch_acc.item())
                a_val_loss.append(a_epoch_loss)

                v_val_accu.append(v_epoch_acc.item())
                v_val_loss.append(v_epoch_loss)
                
                
                ef_val_accu.append(ef_epoch_acc.item())
                ef_val_loss.append(ef_epoch_loss)
                
                lf_val_accu.append(lf_epoch_acc.item())
                lf_val_loss.append(lf_epoch_loss)
            
            print('\n---Audio {} Loss: {:.4f} Acc: {:.4f}'.format(phase, a_epoch_loss, a_epoch_acc))
            print('\n---Video {} Loss: {:.4f} Acc: {:.4f}'.format(phase, v_epoch_loss, v_epoch_acc))
            print('\n---Early Fusion {} Loss: {:.4f} Acc: {:.4f}'.format(phase, ef_epoch_loss, ef_epoch_acc))
            print('\n---Late Fusion {} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, lf_epoch_loss, lf_epoch_acc))
            
            # DEEP COPY THE MODEL
            if phase == 'val':
                save_name = 'models/audio_{}'.format(audio_model)
                save_model('Audio', a_epoch_acc, best_a_acc, save_name, model_a)
                
                save_name = 'models/video_{}'.format(video_model)
                save_model('Video', v_epoch_acc, best_v_acc, save_name, model_v)

                save_name = 'models/ef_{}_{}'.format(audio_model,video_model)
                save_model('EF', ef_epoch_acc, best_ef_acc, save_name, model_ef)
                
                save_name = 'models/lf_lf'
                save_model('LF', lf_epoch_acc, best_lf_acc, save_name, model_lf)

            del dataloader
            torch.cuda.empty_cache()
            gc.collect()

        # EPOCH TIMER
        epoch_time_end = time.time() - epoch_time
        epoch_timer.append(epoch_time_end)
        #time_plot(epoch_timer,'epoch_times','Time per Epoch')

        # plot model data
        plot_data(since, num_epochs, num_videos, learn_rate, b_size, a_train_accu, a_val_accu, a_train_loss, a_val_loss, audio_model, 'audio', best_a_acc)
        plot_data(since, num_epochs, num_videos, learn_rate, b_size, v_train_accu, v_val_accu, v_train_loss, v_val_loss, video_model, 'video', best_v_acc)
        plot_data(since, num_epochs, num_videos, learn_rate, b_size, ef_train_accu, ef_val_accu, ef_train_loss, ef_val_loss, f'{audio_model}_{video_model}', 'early_fusion', best_ef_acc)
        plot_data(since, num_epochs, num_videos, learn_rate, b_size, lf_train_accu, lf_val_accu, ef_train_loss, ef_val_loss, 'lf', 'late_fusion', best_ef_acc)

        if (epoch == (num_epochs-1)):
            plot_cf_matrix(cm_labels,a_predictions,f'results/{audio_model}_cf_{b_size}_{learn_rate}.png')
            plot_cf_matrix(cm_labels,v_predictions,f'results/{video_model}_cf_{b_size}_{learn_rate}.png')
            plot_cf_matrix(cm_labels,ef_predictions,f'results/{audio_model}_{video_model}_cf_{b_size}_{learn_rate}.png')
            plot_cf_matrix(cm_labels,lf_predictions,f'results/lf_cf_{b_size}_{learn_rate}.png')

    # Write results to a .txt file
    write_results(audio_model,video_model,f'{audio_model}_{video_model}',b_size,learn_rate,a_train_accu,a_train_loss,a_val_accu,a_val_loss,v_train_accu,v_train_loss,v_val_accu,v_val_loss,ef_train_accu,ef_train_loss,ef_val_accu,ef_val_loss,lf_train_accu,lf_train_loss,lf_val_accu,lf_val_loss)

    del m, model_a, model_v, model_ef
    gc.collect()
   
    #total_mem, used_mem, free_mem = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])

    #print(f"\n++++++\nRAM - U: {round((used_mem/total_mem)*100,2)}%\n")

if __name__ == '__main__':
    main()
