import os
import csv
import cv2
import numpy as np
import re
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO
from paddleocr import PaddleOCR
from typing import List

# Inicializar FastAPI
app = FastAPI(
    title="API de Reconocimiento de Patentes",
    description="API para detección individual y por lotes de patentes (Portfolio ML)",
    version="2.0"
)

# Cargar modelos una sola vez al iniciar el servidor
print("--- Cargando modelos de IA ---")
modelo = YOLO("runs/detect/train-4/weights/best.pt")
ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)
print("--- Modelos cargados correctamente ---")

def validar_patente(texto):
    """Función de validación flexible para patentes de autos y motos"""
    texto_limpio = texto.replace(" ", "").replace("-", "").replace(".", "").upper()
    
    if len(texto_limpio) not in [6, 7]:
        return None
        
    tiene_letras = any(c.isalpha() for c in texto_limpio)
    tiene_numeros = any(c.isdigit() for c in texto_limpio)
    
    if tiene_letras and tiene_numeros:
        return texto_limpio
    return None

# Estructura de datos para recibir la carpeta por POST
class FolderRequest(BaseModel):
    carpeta: str = "pruebas"

@app.post("/detectar-patente/")
async def detectar_patente(file: UploadFile = File(...)):
    """Endpoint para procesar una sola imagen enviada por HTTP"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if imagen is None:
            return {"error": "No se pudo procesar la imagen enviada."}

        resultados = modelo(imagen)
        candidatos = []

        for r in resultados:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                padding = 5
                h, w, _ = imagen.shape
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                recorte = imagen[y1:y2, x1:x2]
                if recorte.size == 0:
                    continue

                resultado_ocr = ocr.predict(recorte)
                for res in resultado_ocr:
                    textos = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])
                    
                    for texto, score in zip(textos, scores):
                        if texto.strip():
                            patente_valida = validar_patente(texto)
                            if patente_valida:
                                candidatos.append((patente_valida, score))

        if candidatos:
            mejor_texto, mejor_score = max(candidatos, key=lambda x: x[1])
            return {
                "status": "success",
                "patente": mejor_texto,
                "confianza": round(float(mejor_score), 2)
            }
        else:
            return {
                "status": "not_found",
                "mensaje": "No se pudo detectar ninguna patente válida en la imagen."
            }

    except Exception as e:
        return {"status": "error", "detalle": str(e)}

@app.post("/analizar-carpeta/")
async def analizar_carpeta(data: FolderRequest):
    """Endpoint para procesar masivamente todas las imágenes de una carpeta en cualquier ruta"""
    ruta_carpeta = data.carpeta
    
    # --- AQUÍ VA LA VALIDACIÓN ---
    if not os.path.exists(ruta_carpeta):
        return {"error": f"La ruta '{ruta_carpeta}' no existe o no es accesible."}
    
    if not os.path.isdir(ruta_carpeta):
        return {"error": f"La ruta '{ruta_carpeta}' no es una carpeta válida."}
    # -----------------------------
        
    archivo_csv = "resultados_patentes.csv"
    resultados_totales = []
    
    # Listamos solo los archivos de imagen válidos de la ruta recibida
    archivos = [f for f in sorted(os.listdir(ruta_carpeta)) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    with open(archivo_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Imagen', 'Texto_Detectado', 'Confianza'])
        
        for nombre_archivo in archivos:
            # ¡Ojo aquí! Como la carpeta puede estar en otra ruta, 
            # debemos unir la ruta con el nombre del archivo para que OpenCV lo encuentre bien:
            img_path = os.path.join(ruta_carpeta, nombre_archivo)
            
            imagen = cv2.imread(img_path)
            if imagen is None:
                continue
                
            resultados = modelo(imagen)
            candidatos_imagen = []
            
            for r in resultados:
                boxes = r.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    padding = 5
                    h, w, _ = imagen.shape
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)
                    
                    recorte = imagen[y1:y2, x1:x2]
                    if recorte.size == 0:
                        continue
                        
                    resultado_ocr = ocr.predict(recorte)
                    for res in resultado_ocr:
                        textos = res.get('rec_texts', [])
                        scores = res.get('rec_scores', [])
                        for texto, score in zip(textos, scores):
                            if texto.strip():
                                candidatos_imagen.append((texto, score))
                                
            if candidatos_imagen:
                validos = []
                for texto, score in candidatos_imagen:
                    patente_valida = validar_patente(texto)
                    if patente_valida:
                        validos.append((patente_valida, score))
                        
                if validos:
                    mejor_texto, mejor_score = max(validos, key=lambda x: x[1])
                    writer.writerow([nombre_archivo, mejor_texto, f"{mejor_score:.2f}"])
                    resultados_totales.append({
                        "imagen": nombre_archivo, 
                        "patente": mejor_texto, 
                        "confianza": round(float(mejor_score), 2)
                    })
                else:
                    writer.writerow([nombre_archivo, "NO_VALIDO_REGEX", "0.00"])
                    resultados_totales.append({
                        "imagen": nombre_archivo, 
                        "patente": "NO_VALIDO_REGEX", 
                        "confianza": 0.00
                    })
            else:
                writer.writerow([nombre_archivo, "NO_DETECTADO", "0.00"])
                resultados_totales.append({
                    "imagen": nombre_archivo, 
                    "patente": "NO_DETECTADO", 
                    "confianza": 0.00
                })
                
    return {
        "status": "success",
        "carpeta_analizada": ruta_carpeta,
        "total_archivos_procesados": len(resultados_totales),
        "archivo_csv_generado": archivo_csv,
        "resultados": resultados_totales
    }

@app.get("/")
def home():
    return {"mensaje": "API de Patentes activa. Entra a /docs para ver la documentación interactiva."}