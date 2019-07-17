# ======================================================
# Leandro Bugnon (lbugnon@sinc.unl.edu.ar)
# sinc(i) - http://sinc.unl.edu.ar/
# ======================================================
from config_manager import load_config
from test_predict import test_predict

config=load_config("/home/user/src/config")
output_dir=config["output_dir"]
model_dir=config["model_dir"]
test_dir=config["data_dir"]+"test/"

test_predict(test_dir,model_dir,output_dir,config)
