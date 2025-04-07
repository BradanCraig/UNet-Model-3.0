import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np



class MaskDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.Imgs = []
        self.labels = []

        #setting labels train IDS

        self.labels_path = os.path.join("../data", self.root_dir, "masked")#took awa ".., /data, in pathing"
        self.rgb_path = os.path.join("../data", self.root_dir, "raw")
        mask_list = os.listdir(self.labels_path)

        for mask in mask_list:
            self.labels.append(mask)
            tmp = mask.replace("masked","raw")
            self.Imgs.append(tmp)

            

    def __len__(self):
        return len(self.Imgs)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.rgb_path, self.Imgs[index]))
        mask = Image.open(os.path.join(self.labels_path, self.labels[index]))#Getting images at same idx
        
        mask = self.normilize_mask(mask) #classify each pixel [0,3]
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        
        img = transforms.ToTensor()(img)#used to normalize the data and permute [H, W, C] to [C, H, W]

        mask = np.array(mask) #straight conversion
        mask = torch.from_numpy(mask)

        mask = mask.type(torch.LongTensor)
        return img, mask
                             

    def normilize_mask(self, mask):
        mask = np.array(mask)
        
        colormap = {
            '[0, 0, 0]': 0,
            '[255, 0, 0]': 1,
            '[0, 0, 255]': 2, 
            '[0, 255, 0]': 3
        }
        
        w, h = mask.shape[0], mask.shape[1]
        
        vals = [str(list(mask[i,j])) for i in range(h) for j in range(w)]
        new_mask = list([0]*h*w)

        for i, values in enumerate(vals):
            new_mask[i] = colormap[values]#Translation between RGB to Label
        new_mask = np.asarray(new_mask).reshape(h, w)

        return new_mask





