# Clasificación de imágenes de las hojas de cultivos con tres clases (Maíz, calabaza y tomate)
Este es un proyecto de clasificación de imágenes de las hojas de cultivos con tres clases (Maíz, calabaza y tomate). 

 

El dataset con el cuál trabajamos se descargó de Kaggle y tiene como nombre "New Plant Diseases Dataset" el cuál podrán encontrar en el siguiente link: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/ 

Se descargó este dataset y se tomaron 3 clases (Maíz, Calabaza y tomate) estas serán las clases con las que se entrenarán los modelos. 

 

Para empezar, se hicieron los siguientes pasos: 

Eliminación de imágenes corruptas, preparación de datos, división de datos en conjuntos de entrenamientos, validación y prueba, creación de generadores de datos, sintonización de hiperparámetros, entrenamiento del modelo, guardar modelo y visualización de la pérdida durante el entrenamiento. 

  

Luego de esto, se entrenó con los siguientes hiperparámetros: epoch (100), varsize(8), modelname (DenseNet121, EfficientB0 y Resnet101V2), learning rate (1e-2, 1e-3, 1e-4, 1e-5) y optimizer(Adam, SGD, RMSprop). 

  

De los cuales, el mejor modelo fue efficientNetB0 con un learning rate de 1e-3 y el optimizer RMSprop. 

 Luego de obtener el mejor modelo, procedemos a guardarlo y a entrenar el dataset con este. 

Realizando los siguientes pasos: 

Eliminación de imágenes corruptas, preparación de datos, división de datos en conjuntos de entrenamientos, validación y prueba, creación de generadores de datos, entrenamiento del modelo, guardar modelo y visualización de la pérdida durante el entrenamiento. 

  

Con este seguiremos trabajando para crear un servicio web y móvil. 

-Realización servicio web: 

Para esto utilizamos el siguiente código en flask, el cual al ejecutarlo deberá mostrarnos un servicio donde podamos ingresar una imagen de hojas de cultivo y éste la clasifique según la etiqueta que corresponda (Maíz, calabaza o tomate). 

   

-Realización servicio móvil: 

Creamos un servicio móvil por medio de react native, donde luego se descargó expo go en los teléfonos para poder acceder a escanear el QR que nos arroja el código de react native. Al ejecutarse este nos abre la cámara del teléfono y procedemos a tomar foto de hojas para que el modelo pueda clasificarla según corresponda. 
