import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import numpy as np
from sklearn import preprocessing
import os
import ast
from model import MaskModel
from dataset import MaskDataset
from torch.utils.data import DataLoader
from PIL import Image
import torchmetrics




def save_imgs(model, img_h, img_w):
    model.eval()
    colormap = {
        '[0, 0, 0]': 0,
        '[255, 0, 0]': 1,
        '[0, 0, 255]': 2,
        '[0, 255, 0]': 3 
    }

    with torch.no_grad():

        png_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((img_w, img_h))])
        testingData = MaskDataset(root_dir="test", transform=png_transforms)
        testingLoader = DataLoader(testingData, batch_size=1, shuffle=False)#Establishing Test Set and setting the batch size to 1

        for i, (X, y), in (enumerate(testingLoader)):
            #Get predictions for each image in test set

            X, y = X.to('cpu'), y.to('cpu')
            model = model.to('cpu')
            preds = model(X)

            acc_score = IoU(pred=preds, y =y, device = "cpu")
            
            preds = torch.nn.functional.softmax(preds, dim = 1)#logits to preds
            labels = torch.argmax(preds, dim=1)

            print(f"unique labels are {np.unique(labels)}")

            x = transforms.ToPILImage()(X.squeeze(dim=0))

            labels = decode_img(labels, img_h=img_h, img_w=img_w, colormap=colormap)
            y = decode_img(y, img_h=img_h, img_w=img_w, colormap=colormap)
            

            saved_img = Image.new('RGB', (labels.size[0] *3, y.size[1]))
            saved_img.paste(x, (0,0))
            saved_img.paste(y, (labels.size[0], 0))#Saving Raw, Mask, and Pred
            saved_img.paste(labels, (labels.size[0]*2, 0))

            saved_img.save(os.path.join("..", f"results/prediction_{i}_{acc_score}.png"))
    
    print("Images Saved")

def plot_loss(num_of_epochs, vals, title, y_axis):
    epoch_list = range(num_of_epochs)
    for i in vals:
        plt.plot(epoch_list, i)
    plt.xlabel('Epochs')
    plt.ylabel(y_axis)
    plt.title(title)
    plt.savefig(f"../graphs/{title}.png")
    plt.clf()
            

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
    labels_rgb = transforms.RandomHorizontalFlip(p=1)(labels_rgb)
    labels_rgb = torchvision.transforms.functional.rotate(img=labels_rgb, angle=90, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    labels_rgb = transforms.ToPILImage()(labels_rgb.byte())

    return labels_rgb

def IoU(pred, y, device):
    #When you want to see individual class accuracies, the class_iou vairable is a list of all accs
    metric = torchmetrics.JaccardIndex(num_classes=4, task="multiclass").to(device=device)
    class_ious = metric(pred, y)
    mean = torch.mean(class_ious)
    return mean

def calculate_fraction(mask): #Gets Area of Growth
    w, h = mask.size
    total_pixels = h*w
    outputs= []
    
    for k, color in enumerate([[0,0,0], [255,0,0], [0,0,255],[0,255,0]]):
        #Gets percentage coverage for each class
        cmap = np.all(np.equal(mask, color), axis=-1)
        white_pixels = np.count_nonzero(cmap)
        
        if k == 0:
            total_pixels -= white_pixels
        else:
            percentage = white_pixels/total_pixels
            outputs.append(percentage)

    return outputs

def main():
    IMG_H = None
    IMG_W = None
    SIZES = None
    MODEL_NAME = None
    RESULTS_FOLDER = None

    model = MaskModel(inputChannels=3, outputChannels=4, sizes=SIZES)
    model.load_state_dict(torch.load(f"../models/{MODEL_NAME}"), False)

    save_imgs(model, model, IMG_H, IMG_W)

    saved_imgs = os.listdir("../saved_results/")

    for i in range(len(saved_imgs)//2):
        
        img = Image.open(os.path.join(f"../{RESULTS_FOLDER}", saved_imgs[i]))
        y_name = saved_imgs[i].replace("prediction", "truth")
        y = Image.open(os.path.join(f"../{RESULTS_FOLDER}", y_name))
        
        acc = calculate_fraction(img)
        true_acc = calculate_fraction(y)
        mean = IoU(pred=img, y=y)
        
        print(f"{i} has \nNo growth: {acc[0]:.4f}% should be {true_acc[0]:.4f}%\nMedium growth: {acc[1]:.4f}% should be {true_acc[1]:.4f}%\nHigh growth: {acc[2]:.4f}% should be {true_acc[0]:.4f}%\n and accuracy = {mean}")

    print("\n\n\n done")


if __name__ == '__main__':
    main()