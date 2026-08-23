# API de Reconocimiento de Patentes (LPR) 

API desarrollada con FastAPI, YOLO (Ultralytics) y PaddleOCR para la detección y lectura automática de patentes (matrículas) de vehículos a partir de imágenes individuales o procesamiento masivo por lotes.

## Tecnologías Utilizadas
* Python 3.10+
* FastAPI (Framework para la creación de la API REST)
* Ultralytics YOLO (Detección de objetos / localización de la patente)
* PaddleOCR (Reconocimiento óptico de caracteres - OCR)
* OpenCV & NumPy (Procesamiento de imágenes)
* Docker (Contenedorización)

## Endpoints Principales
* `GET /`: Verificación rápida del estado de la API.
* `POST /detectar-patente/`: Recibe un archivo de imagen (`file`) mediante `multipart/form-data`, detecta la patente y devuelve el texto con su nivel de confianza.
* `POST /analizar-carpeta/`: Recibe una ruta en formato JSON (`{"carpeta": "ruta/a/imagenes"}`) y procesa todas las imágenes soportadas (`.png`, `.jpg`, `.jpeg`, `.webp`), generando automáticamente un archivo `resultados_patentes.csv`.

## Ejecución con Docker
La forma más recomendada y aislada de ejecutar esta API es mediante Docker, mapeando los volúmenes locales para que los archivos generados y los pesos de los modelos se compartan con tu máquina.

### 1. Construir la imagen de Docker
Asegúrate de estar en la raíz del proyecto (donde se encuentra el Dockerfile) y ejecuta:

```bash
docker build -t patente-api .

2. Ejecutar el contenedor
Utiliza el siguiente comando para levantar el servicio en el puerto 8001 de tu máquina (mapeado al 8000 del contenedor) y montar tu directorio actual:

Bash
docker run --rm -p 8001:8000 -v $(pwd):/app --name patente-service patente-api
(Nota para Windows PowerShell: si usas PowerShell en lugar de Bash, reemplaza $(pwd) por ${PWD})

Documentación Interactiva
Una vez que el contenedor esté corriendo, puedes acceder a la documentación interactiva generada automáticamente por FastAPI para probar los endpoints:

Swagger UI: http://localhost:8001/docs

ReDoc: http://localhost:8001/redoc