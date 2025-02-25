from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import os


# Configuración
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'  # Carpeta para las imágenes cargadas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# Verificar si la extensión de la imagen es válida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cargar el modelo
model = tf.keras.models.load_model("model/clasificador_basura_model.h5")

# Configurar la carpeta de carga de imágenes
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Función para procesar la imagen
def procesar_imagen(img):
    
    # Redimensionar la imagen al tamaño esperado por el modelo (ajusta según lo que tu modelo necesita)
    img = img.resize((150, 150))  # Ajustamos al tamaño correcto

    img = np.array(img)
    
    # Normalizar la imagen (según cómo tu modelo fue entrenado)
    img = img / 255.0
    
    # Agregar dimensión para que sea compatible con el modelo
    img = np.expand_dims(img, axis=0)
    
    return img

# Ruta para mostrar la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para cargar y predecir la imagen
@app.route('/predict', methods=['POST'])
def predict():
    # Verificar si la solicitud contiene un archivo
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Guardar el archivo subido
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Preprocesar la imagen para el modelo
        img = image.load_img(filename, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalizar la imagen
        
        # Hacer la predicción
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions, axis=1)[0]
        
        # Clases del modelo (de acuerdo al orden en el entrenamiento)
        classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        
        # Retornar la respuesta
        predicted_class = classes[class_idx]
        return jsonify({'predicted_class': predicted_class}), 200
    else:
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    # Crear carpeta uploads si no existe
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Ejecutar el servidor Flask
    app.run(host='0.0.0.0', port=8080, debug=True)

