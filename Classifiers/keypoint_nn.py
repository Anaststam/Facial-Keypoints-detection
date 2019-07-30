import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        self.segmentation = nn.Sequential(
            nn.Conv2d(in_channels = 1,
                      out_channels = 32,
                      kernel_size = 4,
                      stride = 1,
                      
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2
                         ),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels= 32, 
                      out_channels = 64,
                      kernel_size = 3,
                      stride = 1,
        
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2
                         ),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels= 64, 
                      out_channels = 128,
                      kernel_size = 2,
                      stride = 1,
        
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2
                         ),
            nn.Dropout(p=0.3),
            nn.Conv2d(in_channels= 128, 
                      out_channels = 256,
                      kernel_size = 1,
                      stride = 1,
        
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,
                         stride = 2
                         ),
            nn.Dropout(p=0.4),
            
            
      
        )

        self.linear = nn.Sequential(
            nn.Linear(6400, 1000),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 1000),
            nn.Dropout(p=0.6),
            nn.Linear(1000,30)
        )
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        x = self.segmentation(x)
        cnn_output_size = x.size()[1] * x.size()[2] * x.size()[3]
        
        x = x.view(-1, cnn_output_size)

        x = self.linear(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
