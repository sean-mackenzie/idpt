# Dockerfile for IDPT Web Application
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY idpt/ ./idpt/
COPY idpt_web/ ./idpt_web/

# Create storage directory
RUN mkdir -p /app/storage/uploads /app/storage/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV IDPT_WEB_STORAGE_DIR=/app/storage

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "idpt_web.main:app", "--host", "0.0.0.0", "--port", "8000"]
