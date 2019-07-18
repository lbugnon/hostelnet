FROM lbugnon/pytorch-plus-imports

RUN mkdir /home/user/ \
    /home/user/data/ \
    /home/user/data/train/ \
    /home/user/data/test/ \
    /home/user/results/ \
    /home/user/model/ \
    /home/user/src/ 

# src 
COPY src/ /home/user/src/

# data
COPY data/train_labels.csv /home/user/data/ 
ADD data/train.tar.xz /home/user/data/

# Archivos de test. Reemplazar test.tar.xz por el archivo de test final.
# ADD data/test_preliminar.tar.xz /home/user/data
ADD data/test.tar.xz /home/user/data

# Para entrenar todo y generar predicciones sobre test, correr el docker con:
# "python /home/user/src/main.py" 

# Para predecir en nuevos tests con los modelos ya entrenados, se puede construir el docker cambiando solo el archivo test.tar.xz y correr el docker con:
# "python /home/user/src/predict.py"

