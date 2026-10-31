import uvicorn

from .config import settings

if __name__ == "__main__":
    uvicorn.run(
        "pv_assist.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
