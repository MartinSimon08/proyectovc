import os
import csv
import cv2
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR

# --- Función de Validación por Expresiones Regulares Ajustada ---
def validar_patente(texto):
    """
    Limpia espacios, guiones y caracteres raros, y valida 
    que tenga la longitud y estructura lógica de una patente,
    evitando que un espacio o error menor del OCR la tire abajo.
    """
    # 1. Limpieza profunda: sacamos espacios, guiones y cualquier símbolo raro
    texto_limpio = texto.replace(" ", "").replace("-", "").replace(".", "").upper()
    
    # 2. Las patentes argentinas válidas tienen exactamente 6 o 7 caracteres alfanuméricos
    if len(texto_limpio) not in [6, 7]:
        return None
        
    # 3. Verificamos que tenga una combinación lógica de letras y números 
    # (por lo menos alguna letra y algún número)
    tiene_letras = any(c.isalpha() for c in texto_limpio)
    tiene_numeros = any(c.isdigit() for c in texto_limpio)
    
    if tiene_letras and tiene_numeros:
        return texto_limpio
        
    return None
# 1. Cargar modelos
modelo = YOLO("runs/detect/train-4/weights/best.pt")
ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

carpeta_imgs = "pruebas"
if not os.path.exists(carpeta_imgs):
    print(f"La carpeta '{carpeta_imgs}' no existe.")
    exit()

archivo_csv = "resultados_patentes.csv"

# 2. Abrir el archivo CSV
with open(archivo_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Imagen', 'Texto_Detectado', 'Confianza'])

    archivos = sorted(os.listdir(carpeta_imgs))
    print(f"--- Procesando {len(archivos)} imágenes ---")

    for nombre_archivo in archivos:
        if not nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue

        img_path = os.path.join(carpeta_imgs, nombre_archivo)
        print(f"Analizando: {nombre_archivo}")

        imagen = cv2.imread(img_path)
        if imagen is None:
            continue

        resultados = modelo(imagen)
        candidatos_imagen = []

        for r in resultados:
            boxes = r.boxes.xyxy.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # Padding
                padding = 5
                h, w, _ = imagen.shape
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                recorte_patente = imagen[y1:y2, x1:x2]
                if recorte_patente.size == 0:
                    continue

                # OCR
                resultado_ocr = ocr.predict(recorte_patente)
                
                for res in resultado_ocr:
                    textos = res.get('rec_texts', [])
                    scores = res.get('rec_scores', [])
                    
                    for texto, score in zip(textos, scores):
                        if texto.strip():
                            candidatos_imagen.append((texto, score))

        # 3. Decidimos qué escribir en el CSV
        if candidatos_imagen:
            validos = []
            for texto, score in candidatos_imagen:
                patente_valida = validar_patente(texto)
                if patente_valida:
                    validos.append((patente_valida, score))
            
            if validos:
                mejor_texto, mejor_score = max(validos, key=lambda x: x[1])
                writer.writerow([nombre_archivo, mejor_texto, f"{mejor_score:.2f}"])
                print(f"   -> Guardado (Válido): {mejor_texto} ({mejor_score:.2f})")
            else:
                # Si hay texto pero no matchea con la regex estricta
                writer.writerow([nombre_archivo, "NO_VALIDO_REGEX", "0.00"])
                print("   -> Se detectó texto pero no cumplió el formato estricto.")
        else:
            writer.writerow([nombre_archivo, "NO_DETECTADO", "0.00"])
            print("   -> No se detectó nada.")

print(f"\n¡Listo! El archivo '{archivo_csv}' se ha generado correctamente.")