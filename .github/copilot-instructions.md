
# PV-Assist — Project Instructions

## Project objective
Build PV-Assist, an AI-powered technical documentation assistant for photovoltaic maintenance.

The application must allow technicians to ask questions about photovoltaic equipment manuals and receive accurate answers grounded in retrieved document passages.

Every answer should provide document and page references. If the available evidence is insufficient, the assistant must explicitly say so rather than hallucinate.

## Technical stack
- Python 3.12
- FastAPI for the backend
- Streamlit for the frontend
- PyMuPDF for PDF extraction
- FastEmbed for local embeddings
- Qdrant for vector storage and retrieval
- Mistral API for answer generation
- Docker and Docker Compose
- Pytest for automated tests
- Ruff for code quality

## Development principles
- Keep the architecture modular and maintainable.
- Use type hints and clear function names.
- Separate ingestion, retrieval, generation, API and evaluation.
- Do not introduce unnecessary dependencies.
- Do not use LangChain or LlamaIndex unless explicitly requested.
- Keep API keys and secrets in environment variables.
- Never commit .env files or secrets.
- Write tests for important components.
- Explain important architectural decisions.

## User guidance
The developer is an ENSAI engineering student specializing in data science.

Explain technical concepts progressively and clearly.
Do not hide important implementation details behind abstractions.
When proposing changes, explain which files are affected and why.
Do not silently overwrite existing work.
Before running destructive commands or installing major dependencies, request approval.

## Important constraints
- Use the project's .venv environment with Python 3.12.
- Never use the existing PyTorch environment.
- Do not claim that features are implemented unless they have actually been implemented and tested.
- Prioritize retrieval quality, source traceability and answer reliability.
