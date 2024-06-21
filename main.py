from models.cnn_modelMejoradoCuatro import MejoradoCuatroMainTrainModel

if __name__ == "__main__":
    data_dir = './data'
    trainer = MejoradoCuatroMainTrainModel(data_dir)
    trainer.cargar_datos()
    trainer.definir_modelo()
    trainer.entrenar_modelo(epochs=10)

    trainer.guardar_modelo(filename='modelo_clasificacion_autos_camionetas.h5')

    ruta_nueva_imagen = './clasificar/2.png'
    resultado_clasificacion = trainer.clasificar_imagen(ruta_nueva_imagen)
    print("Resultado de la clasificación:", resultado_clasificacion)
