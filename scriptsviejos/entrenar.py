from ultralytics import YOLO

# Cargar un modelo base de YOLOv11
model = YOLO('yolo11n.pt') 

# Entrenar usando la carpeta con el nombre exacto (con mayúsculas)
results = model.train(
    data='PatentesArgentina.yolov11/data.yaml', 
    epochs=40,
    imgsz=500,
    batch=8
)