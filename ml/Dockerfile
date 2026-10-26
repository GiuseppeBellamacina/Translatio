# GPU service of manga mode (docker-compose.manga.yml). Torch with CUDA comes from the base image
# (2.6 at least: manga-ocr's weights need its safe torch.load).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /srv
COPY requirements.txt .
# easyocr pulls the GUI build of OpenCV, which needs X libraries: swap in the headless one
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir opencv-python-headless==5.0.0.93
COPY app.py .

# model weights are downloaded on first use into /models (a volume in compose)
ENV HF_HOME=/models EASYOCR_MODULE_PATH=/models/easyocr DOCTR_CACHE_DIR=/models/doctr
EXPOSE 8100
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8100"]
