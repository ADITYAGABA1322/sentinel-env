FROM node:20-alpine AS ui-builder

WORKDIR /ui

COPY ui/package.json ui/package-lock.json ./
RUN npm ci

COPY ui ./
RUN npm run build


FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY app.py .
COPY environment.py .
COPY models.py .
COPY graders.py .
COPY specialists.py .
COPY trust_ledger.py .
COPY task_graph.py .
COPY comms_bus.py .
COPY mission_context.py .
COPY sentinel_config.py .
COPY difficulty_controller.py .
COPY scenarios.py .
COPY openenv.yaml .
COPY inference.py .
COPY README.md .
COPY pyproject.toml .
COPY server ./server
COPY static ./static
COPY outputs ./outputs
COPY --from=ui-builder /ui/out ./ui/out

# Create outputs directory for baseline scores
RUN mkdir -p outputs

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
