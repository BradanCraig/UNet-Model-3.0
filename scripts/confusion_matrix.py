import torch
from sklearn.metrics import confusion_matrix
from eval import *
from dataset import MaskDataset
import torchvision
from model import MaskModel
from torchvision import transforms as tf
from torch.utils.data import DataLoader
import seaborn as sns
from tqdm import tqdm

#Gets the model and data loaders needed to produce a confusion matrix
def initilize():
    transforms = tf.Compose([tf.ToPILImage(), tf.Resize((2560, 2560))])

    model = MaskModel(inputChannels=3, outputChannels=4, sizes=[16,32,64,128,256,512,1024])
    model.load_state_dict(torch.load("../models/MaskModel_100_Epochs_Fixed_Data_V2"))
    testingData = MaskDataset(root_dir="test", transform=transforms)
    testingLoader = DataLoader(testingData, batch_size=1, num_workers=1, shuffle=False)

    return model, testingLoader

#Creates a plot of the matrix
def plot_matrix(cf):
    class_names = ["Background", "No Growth", "Medium Growth", "High Growth"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cf, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Semantic Segmentation')
    plt.savefig("../graphs/confusion_matrix.png")



def main():     
    model, loader = initilize()
    model.eval()

    preds = []
    truth = []

    for (X,y) in tqdm(loader):
        pred = model(X)
        pred = torch.argmax(pred, dim=1)

        preds.append(pred.view(-1).numpy())
        truth.append(y.view(-1).numpy())

    preds = np.concatenate(preds)
    truth = np.concatenate(truth)#get all predicted and truth values

    cf = confusion_matrix(truth, preds, labels=np.arange(4))

    cf = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
    plot_matrix(cf)

if __name__ == "__main__":
    main()