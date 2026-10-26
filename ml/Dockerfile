# GPU service of manga mode (docker-compose.manga.yml). Torch with CUDA comes from the base image.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

# model weights are downloaded on first start into HF_HOME (a volume in compose)
ENV HF_HOME=/models
EXPOSE 8100
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8100"]
