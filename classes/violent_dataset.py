import cv2 as cv
import gc
import io
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from PIL import Image
from matplotlib import cm, figure
from python_speech_features import mfcc
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

def get_img_from_fig(fig,dpi=180):
    buf = io.BytesIO() #open buffer
    
    fig.savefig(buf, format="png", dpi=dpi,bbox_inches='tight',pad_inches=0) # save figure to buffer

    buf.seek(0) # seek beginning of the buffer
    
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8) # get array from buffer
    
    buf.flush() # flush buffer
    buf.close() # close buffer
    
    img = cv.imdecode(img_arr, 1) # decode image
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # change image color
    img = cv.resize(img, dsize=(224,224), interpolation=cv.INTER_AREA) # resize image
    img = np.swapaxes(img,0,2) # transpose image
    res = torch.from_numpy(img)
    res = res.float()
    return res

def audio_feature_extractor(audio_name):
    plt.axis('off')
    (wav_sr, wav_y) = wavfile.read(audio_name)

    mfcc_feat = mfcc(signal=wav_y,samplerate=wav_sr,nfft=2048)
    mfcc_data = np.swapaxes(mfcc_feat,0,1)

    fig = figure.Figure()
    ax = fig.subplots()
    ax.set_axis_off()
    ax_img = ax.imshow(mfcc_data,interpolation='nearest',cmap=cm.coolwarm,origin='lower',aspect='auto') # generate image
    
    inputs = get_img_from_fig(fig)
    
    del mfcc_feat, mfcc_data
    fig.clf()
    ax.cla() 
    plt.close(fig)
    gc.collect()
    
    return (inputs/255)

def video_feature_extractor(frame):
    # resize image
    im = cv.resize(frame, dsize=(224,224), interpolation=cv.INTER_AREA)

    # BGR to RGB
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

    im_pil = Image.fromarray(im)

    im_pil_transformed = data_transforms(im_pil)

    im_tr = np.asarray(im_pil_transformed)

    im_np = np.asarray(im_pil)

    im_pil.close()

    # to tensor
    batch = torch.from_numpy(im_tr)

    # add dimension to index 0
    batch = torch.unsqueeze(batch, 0)

    # convert to float
    batch = batch.float()

    return batch

num_of_videos = 1

class ViolentDataset(Dataset):
    def __init__(self, root_dir, length=(num_of_videos*2)):
        self.root_dir = root_dir
        self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if (idx < num_of_videos):
            label = 'nv'
            audio_name = f'{self.root_dir}/{label}/{idx}.wav'
            video_name = f'{self.root_dir}/{label}/{idx}.mp4'
        else:
            label='v'
            audio_name = f'{self.root_dir}/{label}/{idx-num_of_videos}.wav'
            video_name = f'{self.root_dir}/{label}/{idx-num_of_videos}.mp4'

        audio_fts = audio_feature_extractor(audio_name)
        audio_fts = torch.Tensor(audio_fts)

        cap = cv.VideoCapture(video_name)
        n_frames = 16

        property_id = int(cv.CAP_PROP_FRAME_COUNT)
        frame_skipper = (int(cv.VideoCapture.get(cap, property_id))//n_frames)-1
        frame_counter = frame_skipper

        extracted_frames = 0
        frame_fts = torch.empty(0,3,224,224)
                
        # VIDEO ANALYSIS
        while(cap.isOpened()):
            ret, frame = cap.read() # iterate frames
            if ret == True:
                if frame_counter == frame_skipper and extracted_frames < n_frames:
                    frame_counter = 0
                    frame_features = video_feature_extractor(frame) # extract features frame
                    frame_fts = torch.cat((frame_fts, frame_features),0) # shape = (n,3,224,224)
                    extracted_frames += 1
                else:
                    frame_counter += 1
            else:
                break

        # When everything done, release 
        # the video capture object
        cap.release()

        # Closes all the frames
        cv.destroyAllWindows()

        frame_fts = frame_fts.permute(1,0,2,3)

        return {'audio': audio_fts, 'label': label, 'video': frame_fts}
