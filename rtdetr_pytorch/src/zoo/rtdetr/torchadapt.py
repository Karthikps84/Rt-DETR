import torch
import torch.nn as nn
from .transformer import *

from src.core import register
from typing import List, Optional, Tuple, Union


__all__ = ['TorchAdapt']

class AdaptorWithMatrix(nn.Module):
    '''
        A different adaptor module which generates a projection of scalar matrix
        instead of a sinlge adaptiveness score to adapt enhancement.
        This matrix will be multiplied with the enhancement to adapt torch (enhancement)
        on each images based on its content.
    '''
    def __init__(self):
        super(AdaptorWithMatrix, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Added Option batch normalization, need to verify
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Added Option batch normalization, need to verify
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # Added Option batch normalization, need to verify
            nn.ReLU()
        )
        self.scalar_matrix = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.scalar_matrix(x) # Output is a matrix with values between 0 and 1

        return x
class Adaptor(nn.Module):
    def __init__(self):
        super(Adaptor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Added Option batch normalization, need to verify
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Added Option batch normalization, need to verify
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # Added Option batch normalization, need to verify
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), # Added dropout for regularization, need to verify
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



class Torch(nn.Module):
    '''
        Switch module:
        Trying different convolutional.
        1- UNet style CNN going from 3->32->64->32 then concat (32,32->24)
        2- Last conv has Residual with SC with output from 1st conv
    '''
    def __init__(self, number_f=32):
        super(Torch, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f*2, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f*2, 24, 3, 1, 1, bias=True)



    def forward(self, x):

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))

        x_r = F.tanh(self.e_conv4(torch.cat([x1, x3], 1)))

        # print(f"orignal x shape : {x.shape} and transformer x_r shape : {x_r.shape}")

        # x_r_resize = F.interpolate(x_r, x.shape[2:], mode='bilinear')
        #
        # print(f"orignal x shape : {x.shape} and transformer x_r shape : {x_r_resize.shape}")

        enhancements = torch.split(x_r, 3, dim=1)

        return enhancements


@register
class TorchAdapt(nn.Module):
    '''
        TorchAdapt module:
        Making a single Unified Module encompassing Torch and Adaptor Modules.
    '''
    #__share__ = ['number_f','adaptor_score' ]
    #__inject__ = ['number_f','adaptor_score' ]
    def __init__(self, number_f=32, adaptor_score=True, loss_color=None, loss_exposure=None,
                 pretrained = True):
        super(TorchAdapt, self).__init__()
        
        checkpoint_path = '/netscratch/hashmi/od/self_supervised_low_light/work_dirs/With_augs_cosine_scheduling_v2_loss_TorchAdapt_r50/epoch_2.pth'
        if pretrained:
            prefix = 'enhancer'
            # Step 1: Load the state dictionary
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            
            # Step 2: Modify the state dictionary to add a prefix to the keys related to the enhancer
            enhanced_state_dict = {}
            for key in state_dict.keys():
                if key.startswith(prefix):
                    # For keys related to the enhancer, add the prefix
                    enhanced_state_dict[f"{prefix}.{key}"] = state_dict[key]
                else:
                    # Otherwise, keep the key unchanged
                    enhanced_state_dict[key] = state_dict[key]
            
            # Step 3: Load the modified state dictionary into the model
            self.load_state_dict(enhanced_state_dict, strict=False)
            print(f'Load enhancer state_dict')
            
        self.torch = Torch(number_f=number_f).cuda()

        # condition to use singe value or a scalar matrix
        self.adaptor_score = adaptor_score
        if self.adaptor_score:
            self.adaptor = Adaptor().cuda()
        else:
            self.adaptor = AdaptorWithMatrix().cuda()


        if loss_color is not None:
            self.loss_color = MODELS.build(loss_color)
        if loss_exposure is not None:
            self.loss_exposure = MODELS.build(loss_exposure)

    def forward(self, x):
        #print('Image enhanced')
        enhancements = self.torch(x)

        adaptiveness = self.adaptor(x)

        # print(f"adaptiveness shape : {adaptiveness.shape}")

        if self.adaptor_score:
            adaptiveness = adaptiveness.view(-1, 1, 1, 1)  # Reshape to match dimensions
            adaptiveness = adaptiveness.expand(-1, 3, x.size(2), x.size(3))  # Expand to match the input image

        # # Initialize final_image with zeros
        final_image = x

        for enhanced_image in enhancements:
            # print(f"enhanced_image shape : {enhanced_image.shape} and adaptiveness shape : {adaptiveness.shape}")
            final_image = final_image + adaptiveness * enhanced_image

            # final_image += enhanced_image

        return final_image, enhancements