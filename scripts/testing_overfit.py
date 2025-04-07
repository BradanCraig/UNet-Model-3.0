import os
import torch
from torch.optim import Adam
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from model import MaskModel
from dataset import MaskDataset
from eval import *
from preprocess import *
from torch.utils.data import DataLoader
from data_augmentation import *

def save_img(X, preds, y, img_h, img_w, num):
    colormap = {
        '[0, 0, 0]': 0,
        '[255, 0, 0]': 1,
        '[0, 0, 255]': 2,
        '[0, 255, 0]': 3 
    }
    preds = torch.nn.functional.softmax(preds, dim = 1)#logits to preds
    labels = torch.argmax(preds, dim=1)

    print(f"unique labels are {np.unique(labels)}")

    X = transforms.ToPILImage()(X.squeeze(dim=0))
    labels = decode_img(labels, img_h=img_h, img_w=img_w, colormap=colormap)
    y = decode_img(y, img_h=img_h, img_w=img_w, colormap=colormap)
    
    saved_img = Image.new('RGB', (labels.size[0] *3, y.size[1]))
    
    saved_img.paste(X, (0,0))
    saved_img.paste(y, (labels.size[0], 0))#Saving Raw, Mask, and Pred
    saved_img.paste(labels, (labels.size[0]*2, 0))
    
    saved_img.save(os.path.join("..", f"prediction_{num}.png"))





def main():

    mask_dir = os.listdir("../data/overfit/masked")
    for name in mask_dir:
        edge_img = fix_edges(cv2.imread(f"../data/overfit/masked/{name}", cv2.IMREAD_COLOR))
        fixed_mask = fix_vals_2(edge_img)
                
        if check_img_vals(fixed_mask):
            raise ValueError(f"Mask {name} is faulty")
        else:
            cv2.imwrite(filename=f"../data/overfit/masked/{name}", img=fixed_mask)


    print("done preprocessing")



    model = MaskModel(inputChannels=3, outputChannels=4, sizes=[16, 32, 64, 128, 256, 512, 1024])
    model.load_state_dict(torch.load("../models/MaskModel_100_Epochs_Fixed_Data_V2"))

    png_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((2048, 2048))])#Can add more later to augment images or might be better to be done in a separate file

    testingData = MaskDataset(root_dir="overfit", transform=png_transforms)
    testingLoader = DataLoader(testingData, batch_size=1, shuffle=False)

    with torch.no_grad():
        testing_acc = 0
        model.eval()
        for i, (X, y) in tqdm(enumerate(testingLoader)):
            X, y = X.to("cpu"), y.to("cpu")
            preds = model(X)
            score = IoU(pred=preds, y=y, device="cpu")
            print(score)
            testing_acc += score
            
            save_img(X=X, preds=preds, y=y, img_h=2048, img_w=2048, num=i)
        testing_acc /= len(testingLoader)

        print(testing_acc)



if __name__ == "__main__":
    main()

