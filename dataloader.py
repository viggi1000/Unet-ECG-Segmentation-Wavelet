
from skimage.transform import resize
from torch.utils.data import Dataset
from skimage import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image depths is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        # assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        
        image, mask, img_name = sample['image'], sample['mask'],sample['image_name']
        image = resize(image, (self.output_size,self.output_size), mode='reflect')
        mask = resize(mask, (self.output_size,self.output_size), mode='reflect')

        return {'image': image, 'mask': mask,'image_name':img_name}
        

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask , img_name = sample['image'], sample['mask'], sample['image_name']
 
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        #mask = mask.transpose((0, 1))    #------- Use this only if output needed is image
        #depth = depth.transpose((2, 0, 1)) 
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask),
                'image_name': img_name}#'depth': torch.from_numpy(depth),


class CustomDataset(Dataset):
    """ dataset."""

    def __init__(self, img_dir, mask_dir, transform=None):
        """
        Args:
            img_dir (string): Path to image directory
            mask_dir (string): Path to masks directory
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_list = os.listdir(self.img_dir)
        self.mask_list = os.listdir(self.mask_dir)


    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        
        # Get the image and mask path
        # Assuming both image and mask folders have name
        image_name = self.img_list[idx]
        image_path = os.path.join(self.img_dir,image_name)
        mask_path  = os.path.join(self.mask_dir,image_name)   

        # Get the image and mask
        image = io.imread(image_path)
        mask  = io.imread(mask_path,as_grey=True)

        sample = {'image': image, 'mask': mask,'image_name':image_name}

        if self.transform:
            sample = self.transform(sample) 

        return sample 
        
# class CrossEntropyLoss2d(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(CrossEntropyLoss2d, self).__init__()
#         self.nll_loss = nn.NLLLoss2d(weight, size_average)

#     def forward(self, logits, targets):
# #         print logits.size()
# #         print targets.size()
#         return self.nll_loss(F.log_softmax(logits), targets)
