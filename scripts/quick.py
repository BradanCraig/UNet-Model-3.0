from eval import *
from model import MaskModel
import time 

#Right now being used to create Lindsays Images
def save_img(preds, img_h, img_w, num):
    
    
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
    
    
    
    colormap = {
        '[0, 0, 0]': 0,
        '[255, 0, 0]': 1,
        '[0, 0, 255]': 2,
        '[0, 255, 0]': 3 
    }
    preds = torch.nn.functional.softmax(preds, dim = 1)#logits to preds
    labels = torch.argmax(preds, dim=1)

    labels = decode_img(labels, img_h=img_h, img_w=img_w, colormap=colormap)
    
    saved_img = Image.new('RGB', (labels.size[0], labels.size[1]))
    saved_img.paste(labels, (0,0))
#Saving Raw, Mask, and Pred

    saved_img.save(f"Lindsey Results/prediction_{num}.png")

def create_figs():
    for i, name in enumerate(os.listdir("Lindseys Images")):
        raw = Image.open(f"Lindseys Images/{name}")
        raw =raw.resize((2048, 2048))

        pred1 = Image.open(f"Lindsey Results/prediction_{(i*3)}.png")
        pred2 = Image.open(f"Lindsey Results/prediction_{(i*3)+1}.png")
        pred3 = Image.open(f"Lindsey Results/prediction_{(i*3)+2}.png")

        final = Image.new("RGB", (raw.size[0]*4, raw.size[1]))
        final.paste(raw, (0,0))
        final.paste(pred1, (raw.size[0], 0))
        final.paste(pred2, (raw.size[0]*2, 0))
        final.paste(pred3, (raw.size[0]*3, 0))
        final.save(f"{name}_fig.png")


def main():
    model = MaskModel(inputChannels=3, outputChannels=4, sizes=[16, 32, 64, 128, 256, 512, 1024])
    model.load_state_dict(torch.load("../models/MaskModel_100_Epochs_Fixed_Data_V2"))
    tran = transforms.Compose([transforms.Resize((2048, 2048)), transforms.ToTensor()])

    start = time.time()
    for i, name in enumerate(sorted(os.listdir("Lindseys Images"))):
        img = Image.open(f"Lindseys Images/{name}")
        img = tran(img)
        img = torch.unsqueeze(img, dim=0)
        for j in range(3):
            pred = model(img)
            save_img(preds=pred, img_w=2048, img_h=2048, num=(i*3)+j)
    end = time.time()
    print((end - start))
    create_figs()

if __name__ =="__main__":
    main()