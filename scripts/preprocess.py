import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm
import ast
import cv2
import matplotlib.pyplot as plt

def unlinkDirs():#Used to clear folders for new datasets
        for dir in os.listdir("data"):
                if dir =="masked" or dir == "raw":
                    pass
                else:
                    dirPath = os.path.join("data/", dir)
                    for file in os.listdir(dirPath):
                        filePath = os.path.join(dirPath, file)
                        os.unlink(filePath)



def normilize_mask(mask):

    colormap = {
        [0,0,0]: 0,
        [255, 0, 0]: 1,
        [0, 0, 255]: 2,
        [0, 255, 0]: 3
    }
    
    #if you switch one, make sure to switch in 'eval.py'

    w, h = mask.shape[0], mask.shape[1]
    
    vals = [str(list(mask[i,j])) for i in range(h) for j in range(w)]
    new_mask = list([0]*h*w)#initializing a list of all 0s

    for i, values in enumerate(vals):
        new_mask[i] = colormap[values]
    
    new_mask = np.asarray(mask).reshape(h, w)

    return new_mask

def splitUncropped(testMaskDir, testRawDir, trainMaskDir, trainRawDir):#Splitting Full Images into Training and Testing
    MASK_DIR = os.path.join(cwd, "..", "data/masked")
    IMG_DIR = os.path.join(cwd, "..", "data/raw")
    
    count=0
    cwd = os.getcwd()

    maskDir = os.listdir(MASK_DIR)
    imgDir = os.listdir(IMG_DIR)

    for i in range(len(maskDir)):
        mask = cv2.imread(os.path.join(MASK_DIR, maskDir[i]), cv2.IMREAD_COLOR)
        rawImg = cv2.imread(os.path.join(IMG_DIR, imgDir[i]), cv2.IMREAD_COLOR)
        
        if count%5 ==0: #80/20 split
            cv2.imwrite(f"{testMaskDir}/{count}_masked.PNG", mask)
            cv2.imwrite(f"{testRawDir}/{count}_raw.PNG", rawImg)
        else:
            cv2.imwrite(f"{trainMaskDir}/{count}_masked.PNG", mask)
            cv2.imwrite(f"{trainRawDir}/{count}_raw.PNG", rawImg) 
        
        print(f"split {count} out of {len(maskDir)}")
        count+=1

def get_num_of_instances():
    #Used for the cross entropy loss weights
    background_count, no_count, med_count, high_count = 0, 0, 0, 0
    mask_dir = sorted(os.listdir("../data/masked"))

    for mask_name in tqdm.tqdm(mask_dir):
        mask = Image.open(f"../data/masked/{mask_name}")

        pixels = mask.load()

        for i in (range(mask.size[0])):
            for j in range(mask.size[1]):
                pix = pixels[i,j]
                if pix == (0,0,0):
                    background_count += 1
                elif pix ==(255,0,0):
                    no_count += 1
                elif pix ==(0,255,0):
                    med_count += 1
                elif pix == (0,0,255):
                    high_count += 1
                else: 
                    raise Exception(f"Found Pixel Value {pix}")
    
    total = background_count+no_count+med_count+high_count
    weights = [(total/(background_count*4)), (total/(no_count*4)), (total/(med_count*4)), (total/(high_count*4))]
    print(weights)
    return weights




def fix_edges(img):
    color_img = img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]#Setting anything with color to 255

    result = np.zeros_like(img)
    contours = cv2.findContours(thresh , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]#Getting just the edge
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if int(area) > 100000:
            cv2.drawContours(result, [cntr], 0, (255,255,255), 1)
            cv2.drawContours(color_img, [cntr], 0, (0,0,0), 2)#Setting Edge to [0,0,0]


    return color_img



def fix_vals_2(img):
    img = cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_NEAREST)

    for i in range(img.shape[0]):#checking for correct pixel values
        for j in range(img.shape[1]):
            
            maxVal = max(img[i, j])
            maxIndex = np.argmax(img[i, j])
            
            if maxVal <200:#Threshhold value has no backing to it
                img[i, j] = (0,0,0)
            else:
                newRGB = []
                
                for k in range(0,3):
                    if k!=maxIndex:
                        newRGB.append(0)
                    else:
                        newRGB.append(255)
                
                img[i, j] = tuple(newRGB)

    return img

def check_img_vals(img):#Checking to make sure that there are only a max for 4 classes
    unique_vals = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    if len(unique_vals)>4:
        print(unique_vals)
        print(len(unique_vals))
        return True
    else:
        return False

def get_rid_of_single_vals(img):
    #Will have to become more robust
    pixels_chanegd = 0
    for i in tqdm(range(img.shape[0]), leave=False):
        for j in range(img.shape[1]):
            try:#boundary conditions
                pix = str(list(img[i, j]))
                Lpix = str(list(img[i, j-1]))
                Rpix = str(list(img[i, j+1]))
                Upix = str(list(img[i+1, j]))
                Dpix = str(list(img[i-1, j]))
            except:
                pass
            
            pixels = [pix, Lpix, Rpix, Upix, Dpix]
            pix_list, pix_count = np.unique(pixels, return_counts=True)
            
            m_idx = np.argmax(pix_count)
            if pix_count[m_idx] > 2:
                img[i, j] = eval(pix_list[m_idx])
                pixels_chanegd+=1
            
            elif pix_count[m_idx] == 2:
                surrounding = [Lpix, Rpix, Upix, Dpix]
                surrounding_list, surrounding_count = np.unique(surrounding, return_counts=True)
                m_idx = np.argmax(surrounding_count)
                img[i, j] = eval(surrounding_list[m_idx])
                pixels_chanegd+=1
            
    return img, pixels_chanegd


def main():
    #Eww
    #refactor and cleaning needed
    
    UNPROCESSED_PATH = "../data/unprocesed"
    pre_existing_masks_num = len(os.listdir("../data/fixed_masks"))
    count = 0

    for dir in tqdm(os.listdir(UNPROCESSED_PATH), desc="Directories", total=(len(os.listdir(UNPROCESSED_PATH)))):
        print("\n", dir)

        for name in tqdm(os.listdir(f"{UNPROCESSED_PATH}/{dir}"), leave=False, desc="Images", total=len(os.listdir(f"{UNPROCESSED_PATH}/{dir}"))):
            
            if "masked" in name.lower() and "unmasked" not in name.lower():
                
                edge_img = fix_edges(cv2.imread(f"{UNPROCESSED_PATH}/{dir}/{name}", cv2.IMREAD_COLOR))
                fixed_mask = fix_vals_2(edge_img)
                
                if check_img_vals(fixed_mask):
                    raise ValueError(f"Mask {name} is faulty")
                else:
                    cv2.imwrite(filename=f"../data/fixed_masks_2/{count}_masked.PNG", img=fixed_mask)
            
            else:
                img = cv2.imread(f"{UNPROCESSED_PATH}/{dir}/{name}", cv2.IMREAD_COLOR)
                img = cv2.resize(img, (2048, 2048))
                cv2.imwrite(filename=f"../data/fixed_raw_2/{count}_raw.PNG", img=img)
                count+=1 
    
    path = "../data/fixed_masks_2"
    pix_change = 0
    for name in tqdm(os.listdir(path)):
        img, num = get_rid_of_single_vals(cv2.imread(os.path.join(path, name), cv2.IMREAD_COLOR))
        pix_change+=num
        cv2.imwrite(f"{path}/{name}", img)
    
    print(f"Pixels Changed on Average = {pix_change/len(os.listdir(path))}")
if __name__ == "__main__":
    main()