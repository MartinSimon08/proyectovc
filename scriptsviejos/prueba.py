from ultralytics import YOLO

# Ajustamos la ruta a la carpeta train-4
modelo = YOLO("runs/detect/train-4/weights/best.pt")

# Asegúrate de que esta imagen exista en la carpeta donde estás ejecutando el script
resultados = modelo("pruebas/prueba500.jpeg", save=True)