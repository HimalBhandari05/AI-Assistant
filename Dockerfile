# Use an official Python 3.12 slim runtime as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables: disable pyc generation and unbuffer logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHROMA_PATH=/app/data/chroma

# Install build essentials if needed for wheels, then clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, UI, and documents
COPY app/ ./app/
COPY ui/ ./ui/
COPY documents/ ./documents/

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start the application with Uvicorn by default
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
