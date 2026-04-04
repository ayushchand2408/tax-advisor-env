FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all environment files
COPY env.py .
COPY baseline_agent.py .
COPY inference.py .
COPY test_env.py .
COPY openenv.yaml .
COPY README.md .

# Default: validate + run inference on all 3 tasks
CMD ["python", "inference.py"]
