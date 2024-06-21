import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from models.cnn_modelMejoradoCuatro import MejoradoCuatroMainTrainModel
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8000", # Agrega cualquier otro origen necesario aquí
    "https://ripe-lizards-buy.loca.lt/"  # Asegúrate de agregar el origen externo aquí
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Inicializar el modelo
data_dir = './data'
trainer = MejoradoCuatroMainTrainModel(data_dir)
trainer.cargar_datos()
trainer.definir_modelo()
trainer.entrenar_modelo(epochs=20)
trainer.guardar_modelo(filename='modelo_clasificacion_autos_camionetas.h5')

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as file:
        return HTMLResponse(content=file.read())

@app.post("/clasificar-imagen/")
async def clasificar_imagen(file: UploadFile = File(...)):
    contents = await file.read()

    # Guardar el archivo temporalmente
    temp_file_path = f'temp_{file.filename}'
    with open(temp_file_path, 'wb') as f:
        f.write(contents)

    # Clasificar la imagen
    resultado_clasificacion = trainer.clasificar_imagen(temp_file_path)

    # Eliminar el archivo temporal
    os.remove(temp_file_path)

    return JSONResponse(content={"resultado": resultado_clasificacion})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
