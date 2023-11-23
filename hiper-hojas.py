import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# balancear las clases
import os
import pandas as pd
import numpy as np
import random
import glob

# Ruta del dataset local
dataset_path = "/home/usco/Documents/dataset-hojas-cultivo"
categories = ["Corn", "Squash", "Tomato"]


num_skipped = 0
for category in categories:
    folder_path = os.path.join(dataset_path, category)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Eliminar imagen corrupta
            os.remove(fpath)

print("Se eliminaron %d imágenes" % num_skipped)




image_size = (224, 224)
batch_size = 8
num_classes = 3
num_channels = 3
input_shape = image_size + (num_channels,)

# Parameters
params = {'dim': image_size,
          'batch_size': batch_size,
          'n_classes': num_classes,
          'n_channels': num_channels,
          'shuffle': False}





image_folder = "/home/usco/Documents/dataset-hojas-cultivo"

Corn_folder = os.path.join(image_folder, "Corn")
Squash_folder = os.path.join(image_folder, "Squash")
Tomato_folder = os.path.join(image_folder, "Tomato")


Corn_path = os.path.join(Corn_folder, "*.JPG")
Squash_path = os.path.join(Squash_folder, "*.JPG")
Tomato_path = os.path.join(Tomato_folder, "*.JPG")


Corn_filenames = glob.glob(Corn_path)
Squash_filenames = glob.glob(Squash_path)
Tomato_filenames = glob.glob(Tomato_path)


print(len(Corn_filenames), len(Squash_filenames),len(Tomato_filenames))



# Determinar el número mínimo de imágenes en una categoría
min_samples = min(len(Corn_filenames), len(Squash_filenames), len(Tomato_filenames))


# Seleccionar aleatoriamente la misma cantidad de imágenes de cada categoría

random.shuffle(Corn_filenames)
random.shuffle(Squash_filenames)
random.shuffle(Tomato_filenames)


Corn_filenames = Corn_filenames[:min_samples]
Squash_filenames = Squash_filenames[:min_samples]
Tomato_filenames = Tomato_filenames[:min_samples]

# Crear un DataFrame con las rutas de las imágenes y sus etiquetas
df_Corn = pd.DataFrame({'filename': Corn_filenames, 'label': 0})
df_Squash = pd.DataFrame({'filename': Squash_filenames, 'label': 1})
df_Tomato = pd.DataFrame({'filename': Tomato_filenames, 'label': 2})


print( len(df_Corn), len(df_Squash), len(df_Tomato))






num_iterations = 7
distribution = {"train":0.70, "val":0.15, "test":0.15}
len_train = int(min_samples * distribution["train"])
len_val = int(min_samples * distribution["val"])
len_test = int(min_samples * distribution["test"])

for i in range(num_iterations):
  # train dataset distribution
  start = i * len_val
  end = start + len_train
  df_train = pd.concat([
       df_Corn[start:end],
       df_Squash[start:end],
       df_Tomato[start:end]
  ])

  if len(df_train) < (len_train*num_classes)-1:
    start = 0
    end = len_train - int(len(df_train)/ num_classes)
    df_train2 = pd.concat([
      df_Corn[start:end],
      df_Squash[start:end],
      df_Tomato[start:end]
    ])

    df_train = pd.concat([df_train, df_train2])

  # print(i, start, end, len(df_train))

  start = end
  if start >= min_samples-1:
    start = 0
  end = start + len_val
  df_val = pd.concat([
      df_Corn[start:end],
      df_Squash[start:end],
      df_Tomato[start:end]
  ])
  # print(i, start, end, len(df_val))

  start = end
  if start >= min_samples-1:
    start = 0
  end = start + len_test
  df_test = pd.concat([
      df_Corn[start:end],
      df_Squash[start:end],
      df_Tomato[start:end]
  ])
  # print(i, start, end, len(df_test))


  train_filename = "train_ds_" + str(i) + ".csv"
  val_filename = "val_ds_" + str(i) + ".csv"
  test_filename = "test_ds_" + str(i) + ".csv"

  df_train = df_train.sample(frac=1)
  df_val = df_val.sample(frac=1)
  df_test = df_test.sample(frac=1)

  df_train.to_csv(train_filename)
  df_val.to_csv(val_filename)
  df_test.to_csv(test_filename)

  # print("-"*60)
  # print()
  
  
  
  

for i in range(num_iterations):
  train_filename = "train_ds_" + str(i) + ".csv"
  val_filename = "val_ds_" + str(i) + ".csv"
  test_filename = "test_ds_" + str(i) + ".csv"

  df_train = pd.read_csv(train_filename)
  df_val = pd.read_csv(val_filename)
  df_test = pd.read_csv(test_filename)

  # print(df_train.groupby(["label"])["label"].count())
  # print(df_val.groupby(["label"])["label"].count())
  # print(df_test.groupby(["label"])["label"].count())
  # print("-"*60)
  # print()






for i in range(num_iterations):
  train_filename = "train_ds_" + str(i) + ".csv"
  val_filename = "val_ds_" + str(i) + ".csv"
  test_filename = "test_ds_" + str(i) + ".csv"

  df_train = pd.read_csv(train_filename)
  df_val = pd.read_csv(val_filename)
  df_test = pd.read_csv(test_filename)

  # print(df_train.groupby(["label"])["label"].count())
  # print(df_val.groupby(["label"])["label"].count())
  # print(df_test.groupby(["label"])["label"].count())
  # print("-"*60)
  # print()
  
    
i = 0
train_filename = "train_ds_" + str(i) + ".csv"
val_filename = "val_ds_" + str(i) + ".csv"
test_filename = "test_ds_" + str(i) + ".csv"

df_train = pd.read_csv(train_filename)
df_val = pd.read_csv(val_filename)
df_test = pd.read_csv(test_filename)

partition = {}
partition["train"] =  list(df_train["filename"])
partition["val"] =  list(df_val["filename"])
partition["test"] =  list(df_test["filename"])

labels = {}
df_all = pd.concat([df_train, df_val, df_test])
for index, row in df_all.iterrows():
  filename = row["filename"]
  label = row["label"]
  labels[filename] = label
# print(labels)








# Importing Image class from PIL module
from PIL import Image

def get_image(image_filename):
    
  # Opens a image in RGB mode
  im1 = Image.open(image_filename).convert("RGB")

  im1 = im1.resize(image_size)
  # print(type(im1))
  image = np.asarray(im1)
  image = np.array(image, dtype='float32')
  image = image /255.0
  # print(image.shape)
  return image



import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=8, dim=(224,224), n_channels=3,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            image_filename = os.path.join(image_folder, ID)
            X[i,] = get_image(image_filename)

            # Store class
            y[i] = self.labels[ID]

            #print(image_filename, y[i])

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    
    
    
# Generators
train_generator = DataGenerator(partition['train'], labels, **params)
val_generator = DataGenerator(partition['val'], labels, **params)
test_generator = DataGenerator(partition['test'], labels, **params)



epochs= 100
callbacks = [
    keras.callbacks.EarlyStopping(monitor='loss', patience=3)
]



import keras_tuner as kt



def model_builder(hp):
  hp_model_type = hp.Choice(
    "model_type",["DenseNet121","EfficientNetB0","ResNet101V2"], default="DenseNet121"
)
    # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice(
      'learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5]
      )

  #optimizer1=keras.optimizers.Adam(learning_rate=hp_learning_rate)
  #optimizer2=keras.optimizers.SGD(learning_rate=hp_learning_rate)
  #optimizer3=keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
  hp_optimizer = hp.Choice(
      "optimizer", ["Adam","SGD","RMSprop"], default = "Adam"
  )

  with hp.conditional_scope("model_type", ["DenseNet121"]):
    if hp_model_type == "DenseNet121":
      model = tf.keras.applications.DenseNet121(
          include_top=True,
          weights=None,
          input_shape=input_shape,
          classes=num_classes,
          classifier_activation="softmax",
      )

  with hp.conditional_scope("model_type", ["EfficientNetB0"]):
    if hp_model_type == "EfficientNetB0":
      model = tf.keras.applications.EfficientNetB0(

          include_top=True,
          weights=None,
          input_shape=input_shape,
          classes=num_classes,
          classifier_activation="softmax",
      )

  with hp.conditional_scope("model_type", ["ResNet101V2"]):
    if hp_model_type == "ResNet101V2":
      model = tf.keras.applications.ResNet101V2(

          include_top=True,
          weights=None,
          input_shape=input_shape,
          classes=num_classes,
          classifier_activation="softmax",
      )

  with hp.conditional_scope("optimizer", ["Adam"]):
    if hp_optimizer == "Adam":
      optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate)

  with hp.conditional_scope("optimizer", ["SGD"]):
    if hp_optimizer == "SGD":
      optimizer=keras.optimizers.SGD(learning_rate=hp_learning_rate)

  with hp.conditional_scope("optimizer", ["RMSprop"]):
    if hp_optimizer == "RMSprop":
      optimizer=keras.optimizers.RMSprop(learning_rate=hp_learning_rate)






#optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
  model.compile(
    optimizer=hp_optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
  )

  return model





tuner = kt.Hyperband(
    model_builder,
    objective='val_accuracy',
    max_epochs=100,
    factor=3,
    overwrite=False,
    directory='/home/usco/Documents/trabajo1-IA-laura-maira/trials',
    project_name='intro_to_kt'
    )





tuner.search(
    train_generator,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_generator,
    )










# Get the optimal hyperparameters
# best_hps=tuner.get_best_hyperparameters(num_trials=3)
best_hps=tuner.get_best_hyperparameters(num_trials=3)
for i in best_hps:
    
    print(f"""
    The hyperparameter search is complete. 
    , optimal learning rate for the optimizer is {i.get('learning_rate')}.
    , optimal model name is {i.get('model_type')}.
    , optimal optimizer name is {i.get('optimizer')}.
    """)










# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=3)


print(best_hps)




# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
#history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

history = model.fit(train_generator,
epochs=100,
callbacks=callbacks,
validation_data=val_generator,
initial_epoch=0
)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))


model_save_path = "/home/usco/Documents/trabajo1-IA-laura-maira/modelcultivo.h5"
model.save(model_save_path)



import matplotlib.pyplot as plt


plt.xlabel("# epoch")


plt.ylabel("Loss Magnitude")


plt.plot(history.history["loss"])