FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install flask joblib numpy scikit-learn
CMD ["python", "app.py"]
