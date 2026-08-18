import os
import csv
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

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
                            # Guardamos todo lo que encuentra el OCR temporalmente
                            candidatos_imagen.append((texto, score))

        # 3. Decidimos qué escribir en el CSV para esta imagen
        if candidatos_imagen:
            # Opcional: si quieres filtrar y guardar el mejor candidato que parezca patente
            # Buscamos uno que tenga al menos 6 caracteres limpios, si no hay, guardamos el de mayor confianza general
            validos = [c for c in candidatos_imagen if len(c[0].replace(" ", "")) >= 6]
            
            if validos:
                mejor_texto, mejor_score = max(validos, key=lambda x: x[1])
            else:
                # Si ninguno cumplió el filtro de 6 caracteres, guardamos el que mayor score tuvo de todos modos
                mejor_texto, mejor_score = max(candidatos_imagen, key=lambda x: x[1])
                
            writer.writerow([nombre_archivo, mejor_texto, f"{mejor_score:.2f}"])
            print(f"   -> Guardado: {mejor_texto} ({mejor_score:.2f})")
        else:
            # Si YOLO no encontró nada o el OCR no leyó absolutamente nada
            writer.writerow([nombre_archivo, "NO_DETECTADO", "0.00"])
            print("   -> No se detectó nada.")

print(f"\n¡Listo! El archivo '{archivo_csv}' se ha generado correctamente con todas las filas.")