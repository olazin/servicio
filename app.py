from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Configuración
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'  # Carpeta para las imágenes cargadas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Verificar si la extensión es válida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cargar el modelo una vez
try:
    model = tf.keras.models.load_model("model/clasificador_basura_model.h5")
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    exit(1)

# Configurar la carpeta de carga de imágenes
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Crear carpeta de imágenes si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Función para preprocesar la imagen
def procesar_imagen(filepath):
    try:
        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizar
        return img_array
    except Exception as e:
        print(f"❌ Error al procesar la imagen: {e}")
        return None

# Página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta de predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró ningún archivo.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo.'}), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Procesar la imagen
        img_array = procesar_imagen(filepath)

        if img_array is None:
            os.remove(filepath)
            return jsonify({'error': 'Error al procesar la imagen.'}), 500

        # Realizar la predicción
        try:
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions, axis=1)[0]
            classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
            predicted_class = classes[class_idx]
        except Exception as e:
            print(f"❌ Error en la predicción: {e}")
            os.remove(filepath)
            return jsonify({'error': 'Error durante la predicción.'}), 500

        # Eliminar la imagen después de la predicción
        os.remove(filepath)

        return jsonify({'predicted_class': predicted_class}), 200

    return jsonify({'error': 'Formato de archivo no válido.'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

