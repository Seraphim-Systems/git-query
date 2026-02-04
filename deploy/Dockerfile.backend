FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .
COPY utils/ ./utils/

# Expose port
EXPOSE 8000

# Run backend
CMD ["python", "app.py"]
