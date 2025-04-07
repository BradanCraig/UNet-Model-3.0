from torchvision.transforms import functional
import albumentations as alb
import torchvision
import os
from PIL import Image
from tqdm import tqdm

def augment_eight(img, path, num):#Must be PIL or Tensor
    #Creates 8 images using Group Theory
    rotated=img
    for i in range(1,4):
        rotated = functional.rotate(img=img, angle=90, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        rotated.save(f"{path}/{num}_rotated_{90*i}.png")
    
    Hflip = torchvision.transforms.RandomHorizontalFlip(p=1)(img)
    Vflip = torchvision.transforms.RandomVerticalFlip(p=1)(img)
    D1Flip = functional.rotate(img=Hflip, angle=90, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    D2Flip =  functional.rotate(img=Hflip, angle=270, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    
    Hflip.save(f"{path}/{num}_HFlipped.png")
    Vflip.save(f"{path}/{num}_VFlipped.png")
    D1Flip.save(f"{path}/{num}_D1flipped.png")
    D2Flip.save(f"{path}/{num}_D2Flipped.png")

def label():
    for i, name in enumerate(os.listdir("../data/augmented_masked")):
        
        mask = Image.open(f"../data/augmented_masked/{name}")
        raw = Image.open(f"../data/augmented_raw/{name}")
        
        mask.save(f"../data/labeled_masks/{i}_masked.png")
        raw.save(f"../data/labeled_raw/{i}_raw.png")



def main():
    label() 


if __name__ == "__main__":
    main()

