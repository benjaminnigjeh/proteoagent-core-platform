FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY memory_store.py server.py ./

ENV DB_PATH=/data/sessions.db

EXPOSE 8001

CMD ["python", "server.py", "--http", "--port", "8001"]
