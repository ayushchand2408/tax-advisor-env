FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY env.py .
COPY app.py .
COPY inference.py .
COPY baseline_agent.py .
COPY test_env.py .
COPY rl_agent.py .
COPY train.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY README.md .
COPY server/ ./server/

EXPOSE 7860

CMD ["python", "server/app.py"]
