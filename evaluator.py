import os
import numpy as np
import tensorflow as tf
from models.cnn_model import ImprovedMainTrainModel

# Define clases para todos los modelos mejorados
from models.cnn_modelMejoradoUno import MejoradoUnoMainTrainModel
from models.cnn_modelMejoradoDos import MejoradoDosMainTrainModel
from models.cnn_modelMejoradoTres import MejoradoTresMainTrainModel
from models.cnn_modelMejoradoCuatro import MejoradoCuatroMainTrainModel

def evaluar_modelo(trainer, epochs=10, filename='modelo_clasificacion.h5', ruta_imagen='./clasificar/2.png'):
    trainer.cargar_datos()
    trainer.definir_modelo()
    trainer.entrenar_modelo(epochs=epochs)
    trainer.guardar_modelo(filename=filename)
    resultado_clasificacion = trainer.clasificar_imagen(ruta_imagen)
    print(f"Resultado de la clasificación ({filename}):", resultado_clasificacion)

if __name__ == "__main__":
    data_dir = './data'

    # Crear instancias de cada modelo mejorado
    modelos = [
        (MejoradoUnoMainTrainModel(data_dir), 'modelo_mejorado_uno.h5'),
        (MejoradoDosMainTrainModel(data_dir), 'modelo_mejorado_dos.h5'),
        (MejoradoTresMainTrainModel(data_dir), 'modelo_mejorado_tres.h5'),
        (MejoradoCuatroMainTrainModel(data_dir), 'modelo_mejorado_cuatro.h5'),
        (ImprovedMainTrainModel(data_dir), 'modelo_mejorado_cinco.h5')
    ]

    # Evaluar cada modelo
    for trainer, filename in modelos:
        print(f"Evaluando {filename}...")
        evaluar_modelo(trainer, epochs=10, filename=filename)
