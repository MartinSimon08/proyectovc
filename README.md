# proyectovc
docker run --rm proyectovc

python3 entrenar.py

python3 -m pip install ultralytics

source .venv/bin/activate
docker run --rm -p 8001:8000 -v $(pwd):/app --name patente-service patente-api