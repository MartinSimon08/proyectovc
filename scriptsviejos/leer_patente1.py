import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# 1. Cargar el modelo YOLO
modelo = YOLO("runs/detect/train-4/weights/best.pt")

# Inicializar PaddleOCR desactivando el motor con conflicto
ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

# 3. Cargar imagen
img_path = "pruebas/prueba500.jpeg"
imagen = cv2.imread(img_path)

# 4. Correr YOLO
resultados = modelo(imagen)

# 5. Recorrer detecciones
for r in resultados:
    boxes = r.boxes.xyxy.cpu().numpy()
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Agregamos un pequeño margen (padding) de 5 píxeles para no cortar los bordes
        padding = 5
        h, w, _ = imagen.shape
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        recorte_patente = imagen[y1:y2, x1:x2]
        
        if recorte_patente.size == 0:
            continue

            # Inferencia con PaddleOCR
        resultado_ocr = ocr.predict(recorte_patente)
        
        # Extraer limpiamente el texto detectado
        for res in resultado_ocr:
            textos = res.get('rec_texts', [])
            scores = res.get('rec_scores', [])
            
            for texto, score in zip(textos, scores):
                if texto.strip():  
                    print(f"--- ¡PATENTE LEÍDA: {texto} (Confianza: {score:.2f}) ---")