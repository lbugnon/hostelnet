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


def test_predict(test_dir,model_dir,output_dir,config):

    
    files=["%s%d.jpg" %(test_dir,f) for f in range(10000) if os.path.isfile("%s%d.jpg" %(test_dir,f))]

    dfres=pd.DataFrame()
    dfres["ind"]=[int(f[f.rfind("/")+1:f.rfind(".")]) for f in files]

    print("Cargando datos de test...")
    test_data=ImData(files,np.zeros(len(files)),config=config)

    # Se cargan los modelos, se generan las predicciones y se promedian.
    allpred=torch.zeros((len(files),16))
    model=Model(output_dir,config=config)
    for m in range(10):
        print("%d/10" %(m+1))
        model.net.load_state_dict(torch.load("%sbest_model_%d.par" %(model_dir,m),map_location=config["device"] ))
    
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=model.nbatch)
    
        prediction=model.run(test_loader,"predict")
        predclass=torch.argmax(prediction,1).cpu().numpy()
    
        allpred+=prediction
             
    predclass=torch.argmax(allpred,1).cpu().numpy()
    with open(output_dir+"prediccion_final.csv","w") as fout:
        for i,p in zip(dfres["ind"],predclass):
            fout.write("%d,%d\n" %(i,p))



    print("listo!")
