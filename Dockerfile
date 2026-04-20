FROM python:3.12-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working dir ───────────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App code ──────────────────────────────────────────────────────────────────
COPY . .

# ── Create required directories ───────────────────────────────────────────────
RUN mkdir -p logs models data

# ── Expose ports ──────────────────────────────────────────────────────────────
# FastAPI
EXPOSE 8000
# Streamlit
EXPOSE 8501

# ── Startup script ────────────────────────────────────────────────────────────
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
