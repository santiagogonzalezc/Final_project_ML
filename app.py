from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# Carga el modelo
with open('models/XGB_RSearchCV.pkl', 'rb') as f:
    modelo = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    output = 2  # Esta es una salida fija para fines de prueba
    return jsonify({'prediction': output})

@app.route('/')
def serve_prediction_page():
    return send_from_directory('static', 'prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
