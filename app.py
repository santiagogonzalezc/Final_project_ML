from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

meses_codificados = {
    "Enero": 2.634,
    "Febrero": 2.121,
    "Marzo": 2.442,
    "Abril": 2.460,
    "Mayo": 3.692,
    "Junio": 3.413,
    "Julio": 2.994,
    "Agosto": 2.453,
    "Septiembre": 1.702,
    "Octubre": 1.617,
    "Noviembre": 1.625,
    "Diciembre": 1.634
}

# Carga el modelo
with open('models/XGB_RSearchCV.pkl', 'rb') as f:
    modelo = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    output = data['input']  

# Codificar la variable
    descripcion_codificada = 0 if output[0] == 'Urgencias pediátricas' else 1
    mes_codificado = meses_codificados.get(output[1].capitalize(), -1)  # Si el mes no está en el diccionario, devuelve -1
    # Codificar la modalidad de contrato (one-hot encoding)
    modalidad_contrato_codificada = [0, 0, 0]  # Inicializa con 3 ceros
    if output[3] == "Capita":
        modalidad_contrato_codificada[0] = 1
    elif output[3] == "Evento":
        modalidad_contrato_codificada[1] = 1
    elif output[3] == "PGP":
        modalidad_contrato_codificada[2] = 1
    # Elimina la variable original de modalidad de contrato
    del output[3]

    # Codificar la régimen de afiliación (one-hot encoding)
    regimen_afiliacion_codificada = [0, 0, 0, 0]  # Inicializa con 4 ceros
    if output[3] == "Contributivo":
        regimen_afiliacion_codificada[0] = 1
    elif output[3] == "Subsidiado":
        regimen_afiliacion_codificada[3] = 1
    elif output[3] == "Especial":
        regimen_afiliacion_codificada[1] = 1
    elif output[3] == "No asegurado":
        regimen_afiliacion_codificada[2] = 1
    # Elimina la variable original que al borrar la anterior ya está en 3
    del output[3]

    # Codificar el tipo de diagnóstico principal
    tipo_diagnostico_codificado = 1 if output[4] == "Repetido" else 0
    # Codificar el sexo
    sexo_codificado = 1 if output[6] == "Femenino" else 0
    # Codificar el triaje
    triaje = output[7]
    if triaje == "NA":
        triaje_codificado = 0
    elif triaje == "5":
        triaje_codificado = 1
    elif triaje == "4":
        triaje_codificado = 2
    elif triaje == "3":
        triaje_codificado = 3
    elif triaje == "2":
        triaje_codificado = 4
    elif triaje == "1":
        triaje_codificado = 5

        # Codificar la causa externa (one-hot encoding)
    causa_externa_codificada = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    causa_externa = output[8]
    if causa_externa == "Accidente de trabajo":
        causa_externa_codificada[0] = 1
    elif causa_externa == "Accidente de tránsito":
        causa_externa_codificada[1] = 1
    elif causa_externa == "Accidente rábico":
        causa_externa_codificada[2] = 1
    elif causa_externa == "Enfermedad general":
        causa_externa_codificada[3] = 1
    elif causa_externa == "Enfermedad profesional":
        causa_externa_codificada[4] = 1
    elif causa_externa == "Lesión por agresión":
        causa_externa_codificada[5] = 1
    elif causa_externa == "Otra":
        causa_externa_codificada[6] = 1
    elif causa_externa == "Otro tipo de accidente":
        causa_externa_codificada[7] = 1
    elif causa_externa == "Sospecha abuso sexual":
        causa_externa_codificada[8] = 1
    del output[8]

    # Extiende la lista de salida con las codificaciones de modalidad de contrato
    output.extend(modalidad_contrato_codificada)
    output.extend(regimen_afiliacion_codificada)
    output.extend(causa_externa_codificada)

    # Ahora reemplazamos la variable original por la codificada
    output[0] = descripcion_codificada
    output[1] = mes_codificado
    output[4] = tipo_diagnostico_codificado
    output[6] = sexo_codificado
    output[7] = triaje_codificado

    return jsonify({'prediction': output})

@app.route('/')
def serve_prediction_page():
    return send_from_directory('static', 'prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
