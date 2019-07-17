# ======================================================
# Leandro Bugnon (lbugnon@sinc.unl.edu.ar)
# sinc(i) - http://sinc.unl.edu.ar/
# ======================================================

from torchvision import transforms
import torch
from torch.autograd import Variable
import torch.nn as nn
import time,random,importlib,pickle
import sklearn.metrics
import numpy as np
import torch.utils.data.sampler
from sklearn.metrics import confusion_matrix

class Model:

    maxepoch=100
    nbatch=8
    W=350

    def __init__(self,out_dir="",config=None,logger=None,fold=None,tmp_dir=""):
    
        net = getattr(importlib.import_module(config["net"].lower()), config["net"])
        
        self.classes=[c for c in config["classes"].split(",")]
                
        self.net=net(self.W,len(self.classes)) 

        self.device=torch.device(config["device"])
        self.net=self.net.to(self.device)  

        self.fold=fold

        # Adam con dos tasas de aprendizaje diferente, más rapido sobre el final de la red
        lr1=1e-5
        lr2=1e-4
        self.optimizer=torch.optim.Adam([{"params": self.net.model.layer1.parameters(),"lr": lr1},{"params": self.net.model.layer2.parameters(),"lr": lr1},{"params": self.net.model.layer3.parameters(),"lr": lr1},{"params": self.net.model.layer4.parameters(),"lr": lr1},{"params": self.net.model.fc.parameters(),"lr": lr2}])        

        self.criterion=torch.nn.CrossEntropyLoss()

        self.partitions=config["partitions"]

        # Early stop config
        self.best_val=np.nan
        self.patience=3
        self.patienceTh=0.01
        self.overfitCount=0

        self.out_dir=out_dir
        self.tmp_dir=tmp_dir

        if logger!=None and fold==0:
            logger.start("params")
            logger.log(str(self.net)+"\n","params")
            logger.log("Nparameters %d\n" %sum(p.numel() for p in self.net.parameters() if p.requires_grad),"params")
        self.logger=logger
        
    def metrics(self,ref,pred,confm=True):

        if type(ref) is not np.ndarray:
            ref=ref.detach().cpu().numpy()
        pred=torch.argmax(pred,1).cpu().numpy()

        uar=sklearn.metrics.recall_score(ref,pred,average="macro")
        acc=sklearn.metrics.accuracy_score(ref,pred)
        balanced_acc=sklearn.metrics.balanced_accuracy_score(ref,pred)
        confmatrix=0
        if confm:
            confmatrix=confusion_matrix(ref,pred,[n for n in range(len(self.classes))]) 
            
        return acc, uar,balanced_acc,confmatrix
        
    def genPartitions(self,files,labels,fold):

        L=len(files)
        if self.partitions=="random":

            ind=np.arange(len(files))
            random.shuffle(ind)

            ntest=max(int(self.partsize[2]*L),1)
            test_ind=ind[:ntest]

            nopt=max(int(self.partsize[1]*L),1)
            optim_ind=ind[ntest:(ntest+nopt)]
          
            train_ind=ind[(ntest+nopt):]
            
        if self.partitions=="xval":
            
            inda=int((fold)*L/nfolds)
            indb=int((fold+1)*L/nfolds)
            
            test_files=files[inda:indb]

            
            trainoptim_files=files[:inda]+files[indb:]
            

            random.shuffle(trainoptim_files)
            Lt=int(.9*len(trainoptim_files))
            train_files=trainoptim_files[:Lt]
            optim_files=trainoptim_files[Lt:]
            
        return train_ind,optim_ind,test_ind

  
    def gen_sampler(self,data=None):
        """Generador para realizar el muestreo de las imagenes. Se define la probabilidad de muestreo de cada clase de forma inversa a la distribución original de los datos."""
        classw = data.class_dist()
        ntrain = data.__len__()
        weights = torch.zeros((ntrain))
        for n in range(ntrain):
            _,l=data.__getitem__(n)
            weights[n]=classw[l]
                
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights),replacement=True)

        return sampler

    
    def set_class_weights(self,w):
        """Encontrar la distribución de las clases a partir de los datos de entrenamiento"""
        
        weight=torch.tensor(w,dtype=torch.float).cuda()
        #weight[2]*=.8
        weight/=torch.sum(weight)
        self.criterion=torch.nn.CrossEntropyLoss(weight=weight)        
    
    def run(self,loader,mode):
        """Correr una época (entrenamiento o test)"""
        if mode=="test" or mode=="predict":
            self.net.eval()
     
        time0=time.time()
        loss=0
        ns=0
        out=None
        ref=None
        for data,labels in loader: # un minibatch (de 8 imágenes) 
            if mode=="train":
                self.optimizer.zero_grad()
                
            bout=self.net(Variable(data))
            bloss=self.criterion(bout,Variable(labels)) # error 

            if mode=="train": # Retropropagación
                bloss.backward()
                self.optimizer.step()
            
            loss+=bloss.item()

            if out is None:
                out=bout.cpu().detach()
                ref=labels.cpu()
            else:
                out=torch.cat((out,bout.cpu().detach()))
                ref=torch.cat((ref,labels.cpu()))
            ns+=1

            
        loss/=ns
        ttime=time.time()-time0

        if mode!="predict":
            acc,uar,balacc,confm=self.metrics(ref,out)

            if mode=="test":
                self.net.train()
            
            return loss,uar,acc,balacc,confm,ttime
        else:
            return out

    def earlystop(self,val,epoch,search_for="max"): 

        if self.best_val==np.nan:
            self.best_val=val
        if (search_for=="min" and val>self.best_val) or (search_for=="max" and val<self.best_val) or abs(self.best_val-val)/self.best_val<self.patienceTh: # comienza overfit
            self.overfit_count+=1
            if self.overfit_count>=self.patience:
                return True
        else:
            self.best_val=val
            self.overfit_count=0
            self.bestepoch=epoch
            torch.save(self.net.state_dict(),"%sbest_model_%d.par" %(self.tmp_dir,self.fold))
            return False
