from datetime import datetime
from sklearn import svm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.video as video

from efmodel import EFModel
from lfmodel import LFModel
from identity import Identity


def get_lf_input(audio_output, video_output, device):
    # Average the predictions out
    stack = torch.cat((audio_output[0],video_output[0]),0) #stack the predictions
    empty = torch.empty((0,4))#.to(device) # create empty tensor in order to create correct shape, in case there is only one prediction
    x = torch.vstack((stack,empty))

    for i in range(1,len(audio_output)):
        x = torch.vstack((x,torch.cat((audio_output[i],video_output[i]),0)))
    
    return x


model_lf = LFModel(4)

for i in range(0,10):

    audio = torch.rand(3,2) # 3 videos

    video = torch.rand(3,2)

    pred = [0,0,0]

    lf_input = get_lf_input(audio,video,None)

    print(model_lf(lf_input))


"""
r18 = models.resnet18()

r18.fc = Identity()

mc3 = models.video.mc3_18()

mc3.fc = Identity()

ef = EFModel(r18, mc3)

output = ef(audio,video)

print(output.shape)
"""

"""
a = torch.Tensor([[0,1,2],[2,1,0],[5,4,3]])
b = torch.Tensor([[0,2,4],[4,2,0],[3,4,5]])
c = torch.empty([3])

mean = torch.mean(torch.vstack((a[0],b[0])),0)

for i in range(1,len(a)):
    mean = torch.vstack((mean,torch.mean(torch.vstack((a[i],b[i])),0)))

print(mean)
"""