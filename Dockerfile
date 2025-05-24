FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .

# Copy app source code
COPY . .

RUN uv sync

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app, listening on all interfaces
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
