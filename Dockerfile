FROM python:3.11-slim
WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir fastapi uvicorn "pydantic>=2.0"

# Copy source code and frozen data
COPY src/ ./src/
COPY data/ ./data/

EXPOSE 7860
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7860"]