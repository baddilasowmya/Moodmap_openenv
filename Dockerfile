FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY moodmap_env/ ./moodmap_env/
COPY graders/ ./graders/
COPY server/ ./server/
COPY app.py .
COPY inference.py .
COPY dashboard.html .
COPY openenv.yaml .
COPY pyproject.toml .

# Expose port
EXPOSE 7860

# Environment variables (API_BASE_URL and MODEL_NAME have defaults; HF_TOKEN does NOT)
ENV API_BASE_URL="<your-active-api-base-url>"
ENV MODEL_NAME="<your-active-model-name>"
# HF_TOKEN must be set at runtime — no default

# Start the FastAPI server using the correct entrypoint
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]