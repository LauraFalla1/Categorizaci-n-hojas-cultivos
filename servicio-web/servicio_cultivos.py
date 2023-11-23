from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
# from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
import numpy as np
import psycopg2
import os

# Configuración para la conexión a la base de datos PostgreSQL
DATABASE = "db_hojas_cultivos"
USER = "neivanet"
PASSWORD = "neivanet"
HOST = "localhost"  # o la dirección IP de tu servidor de base de datos


# Creación de la instancia de Flask
app = Flask(__name__)

# Ruta del modelo preentrenado

MODEL_PATH = 'C:/Users/Lenovo/Music/Trabajo_IA_1/trabajo1-IA-laura-maira/modelhojascultivosv1.h5'
model = load_model(MODEL_PATH)

# Función para conectarse a la base de datos y obtener los nombres de las clases
def get_class_names():
    conn = psycopg2.connect(database=DATABASE, user=USER, password=PASSWORD, host=HOST)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, nombre_category FROM categorys ORDER BY id")
        class_names = {row[0]: row[1] for row in cur.fetchall()}
        
        cur.close()
    finally:
        conn.close()
    return class_names

# Función para preparar la imagen
def prepare_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Función para realizar la predicción
def predict(image_path):
    prepared_image = prepare_image(image_path)
    prediction = model.predict(prepared_image)
    predicted_class = np.argmax(prediction, axis=1)
    class_names = get_class_names()
    predicted_class_name = class_names.get(predicted_class[0])
    return predicted_class_name

@app.route('/', methods=['GET'])
def index():
    # Página principal
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Obtiene el archivo del request
        f = request.files['file']

        # Guarda el archivo en ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Realiza la predicción
        result = predict(file_path)
        
        # Envía el resultado de la predicción
        return result
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=False)