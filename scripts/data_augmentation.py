import albumentations as alb
import os
import cv2
from PIL import Image


#Split the images into 100 tiles
def splitCropped(train_raw_dir, test_raw_dir, train_masked_dir, test_masked_dir):
    #Splitting each image into 100 images

        count=0
        maskDir = os.listdir("../data/masked")
        imgDir = os.listdir("../data/raw")

        for i in range(len(maskDir)):
            mask = Image.open(os.path.join("../data/masked", maskDir[i]))
            rawImg = Image.open(os.path.join("../data/raw", imgDir[i]))
            
            for i in range(10):
                for j in range(10):
                    
                    imgDim = (j * 256, i * 256, (j + 1) * 256, (i +1) * 256)
                    cutMask = mask.crop(imgDim)
                    cutImg = rawImg.crop(imgDim)
                    
                    if count % 5 ==0: # splitting 80% between testing and training
                        cutMask.save(f"{test_masked_dir}/{i}_{j}_{count}_masked.PNG")
                        cutImg.save(f"{test_raw_dir}/{i}_{j}_{count}_raw.PNG")
                    else:
                        cutMask.save(f"{train_masked_dir}/{i}_{j}_{count}_masked.PNG")
                        cutImg.save(f"{train_raw_dir}/{i}_{j}_{count}_raw.PNG")
                count+=1

def main():
    splitCropped(train_raw_dir="../data/new_train/raw", train_masked_dir="../data/new_train/masked", test_raw_dir="../data/new_test/raw", test_masked_dir="../data/new_test/masked")

if __name__ == "__main__":
    main()





