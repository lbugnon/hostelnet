FROM lbugnon/pytorch-plus-imports

RUN mkdir /home/user/ \
    /home/user/data/ \
    /home/user/data/train/ \
    /home/user/data/test/ \
    /home/user/results/ \
    /home/user/model/ \
    /home/user/src/ 

# data
COPY data/train_labels.csv /home/user/data/ 
ADD data/train.tar.xz /home/user/data/

# src 
COPY src/ /home/user/src/

# Archivos de test. Reemplazar test.tar.xz por el archivo de test final.
# ADD data/test_preliminar.tar.xz /home/user/data
ADD data/test.tar.xz /home/user/data

# Para entrenar todo y generar predicciones sobre test
# "python /home/user/src/main.py" 

# Para predecir en test con los modelos ya entrenados, utilizar el comando siguiente
# "python /home/user/src/predict.py"

# ADD models.tar.xz /home/user/models/  # Descomentar esta linea si los modelos se importan desde el host
