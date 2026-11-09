# Image reproductible du pipeline (sans les 45 Go d'images : on monte le dossier).
#   docker build -t chest-xray-pipeline .
#   docker run --rm chest-xray-pipeline                       # échantillon
#   docker run --rm -v "<dossier datasets>:/app/project deep learning datasets" \
#       chest-xray-pipeline python -m src.cli all             # données complètes
FROM python:3.13-slim
WORKDIR /app
ENV PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "src.cli", "all", "--config", "configs/sample.yaml"]
