from app.app import get_rag_answer_impl
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from http import HTTPStatus
from typing import List

from .api_models import RagRequest, RagResponse

import sys
sys.path.append('../../')


models = {}

router = APIRouter()

@router.post("/ask", response_model=RagResponse)
async def predict(request: RagRequest):
    result = await get_rag_answer_impl(request)
    return JSONResponse(content=result[1], status_code=result[0])
