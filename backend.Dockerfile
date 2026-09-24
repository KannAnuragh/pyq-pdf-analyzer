FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-mal \
    fonts-lmodern \
    fonts-texgyre \
    fonts-stix \
    fonts-freefont-ttf \
    fonts-dejavu-core \
    fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt
# Copy the entire backend source code
COPY . /app/pdf_qa_agent

# Remove frontend from backend image to save space
RUN rm -rf /app/pdf_qa_agent/frontend

# Set environment variables for persistent storage
ENV DATA_DIR="/app/pdf_qa_agent/data"
ENV SQLALCHEMY_DATABASE_URL="sqlite:////app/pdf_qa_agent/data/pdf_qa.db"

CMD ["uvicorn", "pdf_qa_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
