# ======================================================
# Leandro Bugnon (lbugnon@sinc.unl.edu.ar)
# sinc(i) - http://sinc.unl.edu.ar/
# ======================================================
import shutil,os,sys,random,time
from torch.autograd import Variable
import torch
import numpy as np
from model import Model
import pandas as pd
from logger import Logger
from config_manager import load_config
from imData import ImData
from test_predict import test_predict

# Reproducibilidad (dentro de lo que permite la variabilidad en GPUs) ==
seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# ======================================================================

# Parametros globales ===================================================
config=load_config("/home/user/src/config")
Nfolds=int(config["nfolds"])
train_dir=config["data_dir"]+"train/"

config["partitions"]=[float(p) for p in config["partitions"].split(",")]

output_dir=config["output_dir"]
shutil.rmtree(output_dir,ignore_errors=True)
try:
    os.mkdir(output_dir)
except:
    nada=0
model_dir=config["model_dir"]
shutil.rmtree(model_dir,ignore_errors=True)
os.mkdir(model_dir)    
# =================================================================

logger=Logger(output_dir)
logger.start("res")
logger.log("test_uar,test_balanced_acc,bestepoch\n")

# Etiquetas y archivos de train ===================================
df=pd.read_csv(config["data_dir"]+"train_labels.csv")
files=np.array(["{}{}.jpg".format(train_dir,i) for i in df["fileName"]])
labels=np.array(df["tag"].values).astype(int)

Nsamples=int(config["nsamples"])

for fold in range(Nfolds):

    logger.start("train")
    print("Train fold %d/%d" %(fold,Nfolds))
    
    print("Cargando imagenes en RAM para facilitar entrenamiento (lleva un par de minutos)")
    
    ind=np.arange(Nsamples)
    np.random.shuffle(ind)

    l=int(len(ind)*config["partitions"][0])
    m=l+int(len(ind)*config["partitions"][1])

    train_ind=ind[:l]
    
    train_data=ImData(files[train_ind],labels[train_ind],config=config,augment=int(config["augment"]),objlabels=True)
    # Model initialization
    model=Model(output_dir,config=config,logger=logger,tmp_dir=model_dir,fold=fold)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=model.nbatch, sampler=model.gen_sampler(train_data))

    if config["partitions"][1]>0:
        optim_ind=ind[l:m]
        optim_data=ImData(files[optim_ind],labels[optim_ind],config=config)
        optim_loader = torch.utils.data.DataLoader(optim_data, batch_size=model.nbatch)
    if config["partitions"][2]>0:
        test_ind=ind[m:]
        test_data=ImData(files[test_ind],labels[test_ind],config=config)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=model.nbatch)

    model.set_class_weights(train_data.class_dist())

    logger.log("epoch,train_loss,train_uar,optim_loss,optim_uar,optim_acc,optim_balanced_acc,train_time\n")
    print("epoch,tloss,tuar,oloss,ouar,oacc,obalacc,ttime\n")
    optim_loss,optim_uar,optim_acc,optim_bal_acc=0,0,0,0
    for epoch in range(model.maxepoch):

        train_loss,train_uar,train_acc,_,_,epoch_time=model.run(train_loader,"train")

        
        if config["partitions"][1]>0:
            optim_loss,optim_uar,optim_acc,optim_bal_acc,optim_confm,testtime=model.run(optim_loader,"test")
        
        logstr="%3d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f\n" %(epoch,train_loss,train_uar,optim_loss,optim_uar,optim_acc,optim_bal_acc,epoch_time)
        logger.log(logstr,"train")
        print(logstr)
    
        if epoch>1 and config["partitions"][1]>0:
            if model.earlystop(optim_loss,epoch,search_for="min"): 
                break
        best_epoch=epoch
        
    # Se carga el modelo en el punto óptimo (según lo estimado en earlystop.
    if config["partitions"][1]>0:
        best_epoch=model.bestepoch
        model.net.load_state_dict(torch.load("%sbest_model_%d.par" %(model_dir,fold)))
        del optim_data
    else:
        torch.save(model.net.state_dict(),"%sbest_model_%d.par" %(model_dir,fold))
            
    del train_data
    
    if config["device"]==torch.device("cuda"):
        torch.cuda.empty_cache()
    if config["partitions"][2]>0:
        _,test_uar,test_acc,test_bal_acc,test_confm,_=model.run(test_loader,"test")
        logger.log("%.4f,%.4f,%.4f,%d\n" %(test_uar,test_acc,test_bal_acc,best_epoch),"res")
        logger.log(str(test_confm)+"\n","confm")
        del test_data
    del model
    if config["device"]==torch.device("cuda"):
        torch.cuda.empty_cache()

logger.close()


# Prediccion ===============================================================================
# ==========================================================================================
# ==========================================================================================

test_dir=config["test_dir"]

test_predict(test_dir,model_dir,output_dir,config)
