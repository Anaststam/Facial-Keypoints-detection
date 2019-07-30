import torch
from torchvision import transforms, utils
import numpy as np
# tranforms


class Normalize(object):
    """Normalizes keypoints.
    """
    def __call__(self, sample):
        
        image, key_pts = sample['image'], sample['keypoints']
#         image=image.reshape(96,96)
        
#         rows,columns = image.shape
        
#         rows2,columns2 = key_pts.shape
        
        ##############################################################
        # TODO: Implemnet the Normalize function, where we normalize #
        # the image from [0, 255] to [0,1] and keypoints from [0, 96]#
        # to [-1, 1]                                                 #
        ##############################################################
#         maxvalue=np.amax(image)
#         minvalue=np.amin(image)
#         for i in range (rows):
#             for j in range (columns):
#                 image[i][j]=(image[i][j] - minvalue) / (maxvalue-minvalue)
#         image=image.reshape(1,rows,columns)
        
#         maxvalue2=np.amax(key_pts)
#         minvalue2=np.amin(key_pts)
#         for i2 in range(rows2):
#             for j2 in range(columns2):
#                 key_pts[i2][j2]=(2*(key_pts[i2][j2] - minvalue2) / (maxvalue2-minvalue2)) -1
        
        image=image/255
        key_pts= (key_pts/(96/2))-1
        
        ##############################################################
        # End of your code                                           #
        ##############################################################
        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        return {'image': torch.from_numpy(image).float(),
                'keypoints': torch.from_numpy(key_pts).float()}