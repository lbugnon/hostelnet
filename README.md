# Metadata Despegar 19: HostelNet

Este repositorio contiene el código fuente para reproducir los resultados enviados a la competencia.

## Construcción del docker
Para entrenar los modelos y generar predicciones, descargar el repositorio y copiar los datos comprimidos a la carpeta "data/".

```bash
unzip hostelnet-master.zip 
mv train.tar.xz test.tar.xz train_labels.csv hostelnet-master/data
```
Y construir el docker
```bash
cd hostelnet-master/
docker build -t hostelnet .
```
El docker contiene todos los paquetes necesarios de python y el CUDA dev kit, pero requiere de los drivers de Nvidia en el host. De momento la solución más simple y con un rendimiento similar a usar el codigo directamente en el host parece ser instalar nvidia-docker2 (https://github.com/NVIDIA/nvidia-docker), que debería ser indiferente al la placa y driver instalado (compatible con CUDA9.0).

## Entrenamiento y validación

Para entrenar el modelo final (validación y predicción en test incluido): 
```bash
docker run --runtime=nvidia -it -v /ABSOLUTE_PATH/results:/home/user/results/ hostelnet python3 /home/user/src/main.py
```

En la carpeta "./results" (del host) se guardarán los logs de entrenamiento, los modelos entrenados y la prediccion sobre los datos de test. 

## Predicción
Con los modelos ya entrenados, se puede construir el docker con otros datos de test y realizar solo la predicción con:
```bash
docker run --runtime=nvidia -it -v /ABSOLUTE_PATH/results:/home/user/results/ hostelnet python3 /home/user/src/predict.py
```
