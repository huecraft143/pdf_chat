FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bake embedding model into the image — avoids downloading it on every cold start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY main.py main.py

ENV CHROMA_DIR=/tmp/chroma_db

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
