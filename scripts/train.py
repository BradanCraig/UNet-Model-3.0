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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#Hyperparameters
IMG_HEIGHT = 2048
IMG_WIDTH = 2048
BATCH_SIZE = 1
NUM_WORKERS = 4
EPOCHS = 100
LEARNING_RATE = .0001

MODEL_PATH = os.path.join("..", "models")   
MODEL_NAME = "MaskModel_100_Epochs_Fixed_Data_V1"


def train_epoch(dataloader, model, optimizer, loss_fn):
    training_loss = 0
    training_acc = 0
    model.train()
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X, y = batch
        X, y = X.to(DEVICE), y.to(DEVICE)
        preds = model(X)
        loss = loss_fn(preds, y)
        
        training_loss += loss
        training_acc += IoU(pred=preds, y=y, device=DEVICE)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    training_loss /= len(dataloader)
    training_acc /= len(dataloader)
    return training_loss, training_acc

def test_epoch(dataloader, model, loss_fn):
    with torch.no_grad():
        testing_loss = 0
        testing_acc = 0
        model.eval()
        for i, (X, y) in tqdm(enumerate(dataloader)):
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X)
            loss = loss_fn(preds, y)
            testing_loss+=loss
            testing_acc += IoU(pred=preds, y=y, device=DEVICE)
        testing_loss /= len(dataloader)
        testing_acc /= len(dataloader)
    return testing_loss, testing_acc
            


def main():

    train_loss_vals = []
    test_loss_vals = []
    test_acc_vals = []
    training_acc_vals = []
    png_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((IMG_HEIGHT, IMG_WIDTH))])#Can add more later to augment images or might be better to be done in a separate file
    
    
    trainingData = MaskDataset(root_dir="train", transform=png_transforms)
    testingData = MaskDataset(root_dir="test", transform=png_transforms)

    trainingLoader = DataLoader(trainingData, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    testingLoader = DataLoader(testingData, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    model = MaskModel(inputChannels=3, outputChannels=4, sizes=[16, 32, 64, 128, 256, 512, 1024]).to(device=DEVICE)
    optimizer = Adam(params=model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nstarting training on {DEVICE}\n")
    for epoch in range(EPOCHS):
        train_loss, train_acc= train_epoch(dataloader=trainingLoader, model=model, optimizer=optimizer, loss_fn=loss_fn)
        test_loss, test_acc= test_epoch(dataloader=testingLoader, model=model, loss_fn=loss_fn)
        
        train_loss_vals.append(train_loss.item())
        test_loss_vals.append(test_loss.item())
        training_acc_vals.append(train_acc.item())
        test_acc_vals.append(test_acc.item())
        
        if epoch % 10 == 0:
            save_imgs(dataloader=testingLoader, model=model, img_h=IMG_HEIGHT, img_w=IMG_WIDTH, device=DEVICE, epoch = epoch)
            torch.save(obj=model.state_dict(), f=os.path.join(MODEL_PATH, f"{MODEL_NAME}_{epoch}" ))
        
        print(f"Epoch {epoch} finished\n\nTraining Loss {train_loss} \nTesting Loss {test_loss} \nTraining Acc {train_acc} \nTesting Acc {test_acc}")
    
    print("Training is complete")
    torch.save(obj=model.state_dict(), f=os.path.join(MODEL_PATH, MODEL_NAME))
    
    print(f"Model Saved")
    plot_loss(num_of_epochs=EPOCHS, vals=[train_loss_vals, test_loss_vals], title=f"Loss Over Epoch {MODEL_NAME}", y_axis="Loss")
    plot_loss(num_of_epochs=EPOCHS, vals=[training_acc_vals, test_acc_vals], title= f"Accuracy Over Epoch {MODEL_NAME}", y_axis="Accuracy")
    save_imgs(dataloader=testingLoader, model=model, img_h=IMG_HEIGHT, img_w=IMG_WIDTH, device=DEVICE)
    


if __name__ == '__main__':
    main()





