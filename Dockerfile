# Dockerfile
FROM python:3.12.2-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -U -r requirements.txt

COPY . .

# Run the training script to generate model.joblib and scaler.joblib
RUN python src/train.py

EXPOSE 5000

# Run main.py when the container launches
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000","--reload"]