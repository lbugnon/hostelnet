# ======================================================
# Leandro Bugnon (lbugnon@sinc.unl.edu.ar)
# sinc(i) - http://sinc.unl.edu.ar/
# ======================================================
from torch.utils.data import Dataset
from torchvision import transforms
import torch,os
import numpy as np
from PIL import Image
import torchvision, pickle
import pandas as pd

class ImData(Dataset):
    """Modelo de datos y transformaciones que utilizará el clasificador"""

        
    def __init__(self, file_list,label_list,config,im_size=[350,350],logger=None,augment=0): 

        self.augment=augment
        
        self.classes=[c for c in config["classes"].split(",")]
        self.device=config["device"]
        self.im_size=im_size
   
        # El tamaño del dataset permite subir todo a RAM 
        self.data=torch.zeros((len(file_list),3,im_size[0],im_size[1]))
        self.labels=-torch.ones((len(file_list)),dtype=torch.long)
       
        for k in range(len(file_list)):
            data=self.load_file(file_list[k])
            self.data[k,:,:,:]=data.cpu()
            self.labels[k]=int(label_list[k])

        self.data=self.data[self.labels>=0,:,:,:]
        self.labels=self.labels[self.labels>=0]
     
    def __len__(self):
        return self.data.shape[0]

    def class_dist(self):

        w=[len(np.where(self.labels==c)[0])/len(self.labels) for c in range(len(self.classes))]

        return max(w)/np.array(w)
    
    def baseTransform(self,im):
        """Normalización de los tamaños de imagen para levantar las imágenes"""

        transform = transforms.Compose([transforms.Resize((self.im_size[0],self.im_size[1])),transforms.ToTensor()])
        im=transform(im)
       
        return im


    def augmentTransform(self,im):
        """Aumentado de datos: variaciones aleatorias de brillo, intensidad y tono, espejado y rotaciones. Normalización del tensor final"""

        if self.augment:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(hue=.05, brightness=.1,contrast=.1),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20, resample=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        return transform(im)

    
    def load_file(self,file_name):
        """Carga de archivos con correcciones para archivos con fallas o formatos diferentes (se elimina el canal alfa)"""
        
        #print(file_name)
        im=Image.open(file_name)
        
        try:
                
            if im.mode!="RGB":
                im=im.convert("RGB")
        
            return self.baseTransform(im).to(self.device)
        except:
            
            os.system("convert -quiet %s /home/user/results/tmp.jpg" %file_name)
            im=Image.open("/home/user/results/tmp.jpg")
            
            if im.mode!="RGB":
                im=im.convert("RGB")

                
            return self.baseTransform(im).to(self.device)


    def __getitem__(self,i):
        """Devolución de un elemento del dataset."""
        return self.augmentTransform(self.data[i,:,:,:]).to(self.device),self.labels[i].to(self.device)
        
                    
         

