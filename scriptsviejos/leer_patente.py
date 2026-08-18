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
        recorte_patente = imagen[y1:y2, x1:x2]
        
        if recorte_patente.size == 0:
            continue

        # 6. USAR .predict() EN LUGAR DE .ocr() PARA LA NUEVA VERSIÓN
        resultado_ocr = ocr.predict(recorte_patente)
        
        # 7. Extraer e imprimir el texto
        if resultado_ocr:
            print("\n--- ¡PATENTE DETECTADA! ---")
            for res in resultado_ocr:
                # Dependiendo de la estructura del output de predict en esta versión:
                print(res)