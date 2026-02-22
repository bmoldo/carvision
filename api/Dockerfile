FROM python:3.12-slim

WORKDIR /app

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY sdk/python /app/sdk/python
COPY api/server.py /app/server.py

# Models are mounted as a volume at runtime
VOLUME /app/models

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
