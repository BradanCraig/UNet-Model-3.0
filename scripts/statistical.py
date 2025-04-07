import statistics
import torch
import torchvision as tv
from torchmetrics import JaccardIndex
from model import MaskModel
from dataset import MaskDataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
import numpy as np
from eval import save_imgs
import ast
import matplotlib.pyplot as plt

device = "cpu" if torch.cuda.is_available() == False else "cuda"

def save_img(preds, y, img_h, img_w, num):
    
    
    def decode_img(img, img_h, img_w, colormap):
        reverse_colormap = {v: k for k, v in colormap.items()}

        # Apply the reverse colormap to convert integers to RGB values
        # Using list comprehension to map each label
        flattened_labels = img.flatten()
        labels_rgb_flatten = ([reverse_colormap[label.item()] for label in flattened_labels])
        
        labels_rgb_flatten= [ast.literal_eval(x) for x in labels_rgb_flatten]

        labels_rgb = torch.tensor(labels_rgb_flatten).reshape((img_h, img_w, 3))
        labels_rgb = labels_rgb.permute(2, 1, 0)

        labels_rgb = transforms.Resize((img_h, img_w))(labels_rgb)
        labels_rgb = transforms.ToPILImage()(labels_rgb.byte())

        return labels_rgb
    
    
    
    colormap = {
        '[0, 0, 0]': 0,
        '[255, 0, 0]': 1,
        '[0, 0, 255]': 2,
        '[0, 255, 0]': 3 
    }
    preds = torch.nn.functional.softmax(preds, dim = 1)#logits to preds
    labels = torch.argmax(preds, dim=1)

    labels = decode_img(labels, img_h=img_h, img_w=img_w, colormap=colormap)
    y = decode_img(y, img_h=img_h, img_w=img_w, colormap=colormap)
    
    saved_img = Image.new('RGB', (labels.size[0] *2, y.size[1]))
    saved_img.paste(y, (0,0))
    saved_img.paste(labels, (labels.size[0], 0))#Saving Raw, Mask, and Pred

    saved_img.save(os.path.join("stats_results", f"prediction_{num}.png"))



def get_class_accs(X, y, num): 
    metric_multiclass = JaccardIndex(num_classes=4, task="multiclass", average=None)
    metric = JaccardIndex(num_classes=4, task="multiclass")
    
    model = MaskModel(inputChannels=3, outputChannels=4, sizes=[16, 32, 64, 128, 256, 512, 1024])
    model.load_state_dict(torch.load("../models/MaskModel_100_Epochs_Fixed_Data_V1"))
    model.eval()

    pred = model(X)
    save_img(num=num, img_h=2048, img_w=2048, preds=pred, y=y)
    return metric(pred, y), metric_multiclass(pred,y)


def get_test_imgs():
    model = MaskModel(inputChannels=3, outputChannels=4, sizes=[16, 32, 64, 128, 256, 512, 1024])
    model.load_state_dict(torch.load("../models/MaskModel_100_Epochs_Fixed_Data_V1"))
    save_imgs(model=model, img_h=2048, img_w=2048)



def plot(vals):
    plt.hist(vals, bins=25)
    plt.savefig("../graphs/Plot of Test Split Accuracy.PNG")
    plt.clf()











def main():
    png_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((2048, 2048))])
    testingData = MaskDataset(root_dir="test", transform=png_transforms)
    testingLoader = DataLoader(testingData, batch_size=1, shuffle=False)

    img_vals = []
    background_acc, low_acc, mid_acc, high_acc = 0, 0, 0, 0
    for i, (X, y) in enumerate(tqdm(testingLoader)):
        overall_acc, class_accs = get_class_accs(X=X, y=y, num=i)
        img_vals.append(overall_acc.item())
        
        class_accs = class_accs.tolist()
        class_accs = [x if x != 0 else 1 for x in class_accs]#Returns 0 if class isn't in image

        background_acc+=class_accs[0]
        low_acc +=class_accs[1]
        mid_acc +=class_accs[2]
        high_acc +=class_accs[3]

    background_acc /= len(testingLoader)
    low_acc /= len(testingLoader)
    mid_acc /= len(testingLoader)
    high_acc /= len(testingLoader)
    
    print(f"Min = {min(img_vals)}, Max = {max(img_vals)}, Var = {statistics.variance([x*100 for x in img_vals])}, Mean = {statistics.mean(img_vals)}")
    print(f"Accuracy Along Test Classes\nBackground: {background_acc}\nLow: {low_acc}\nMid: {mid_acc}\nHigh: {high_acc}")
    plot(vals=img_vals)





if __name__ == "__main__":
    main()