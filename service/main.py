import uvicorn
from api.v1.api_route import router
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import logging

from app.app import load_models, load_qdrant
# from transformers import VectorizeTransformer, TfidfVectorizeTransformer


load_models()
load_qdrant()

app = FastAPI(
    title="rag",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=False)
