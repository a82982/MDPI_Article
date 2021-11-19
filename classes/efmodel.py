import torch
import torch.nn as nn

class EFModel(nn.Module):
    def __init__(self,modelA, modelV, device):
        super(EFModel, self).__init__()
        self.modelA = modelA
        self.modelV = modelV
        self.device = device
        self.fc = nn.Sequential(
                            nn.Dropout(p=0.25),
                            nn.Linear(512,2)
                        )

    def forward(self, audio, video):
        audio_output = self.modelA(audio) # predict from audio
        video_output = self.modelV(video) # predict from video
        
        # Average the predictions out
        stack = torch.vstack((audio_output[0],video_output[0])) #stack the predictions
        mean = torch.mean(stack,0) # mean the stacked predictions
        empty = torch.empty((0,512)).to(self.device) # create empty tensor in order to create correct shape, in case there is only one prediction
        x = torch.vstack((mean,empty))

        for i in range(1,len(audio_output)):
            x = torch.vstack((x,torch.mean(torch.vstack((audio_output[i],video_output[i])),0)))
        
        x = self.fc(x) # pass through classifier
        return x
        