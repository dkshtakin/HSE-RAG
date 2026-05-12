from api.v1.api_models import RagRequest, RagResponse
# from copy import deepcopy
import os
import asyncio
import logging
# from joblib import dump, load
from fastapi import HTTPException
import time
# from concurrent.futures import ProcessPoolExecutor
# from dotenv import load_dotenv
# import re
# import joblib
import sys
sys.path.append('../')

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
)
from tqdm.auto import tqdm
from llama_cpp import Llama

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

from huggingface_hub import snapshot_download, hf_hub_download


import gc
import torch
import numpy as np

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(DEVICE, torch.__version__)

FILTER_MODEL_NAME = 'dkshtakin/RuBioRoBERTa-RAG-filter'
FILTER_MODEL_DIR = 'models/RuBioRoBERTa-RAG-filter'
EMBEDS_MODEL_NAME = 'google/embeddinggemma-300m'
EMBEDS_LOCAL_DIR = 'embeddings/embeddinggemma-300m'
SPARSE_EMBEDS_NAME = 'Qdrant/BM25'
RERANK_MODEL_NAME = 'jinaai/jina-reranker-v3'
RERANK_LOCAL_DIR = 'models/jina-reranker-v3'
LLM_MODEL_REPO = 'unsloth/Qwen3.5-4B-GGUF'
LLM_MODEL_NAME = 'Qwen3.5-4B-Q4_0.gguf'
LLM_LOCAL_DIR = f'models/{LLM_MODEL_NAME}'

CHUNK_SIZE = 1280
CHUNK_OVERLAP = 16
HYBRID = True
SUFFIX = '_hybrid'

TEMPLATE = """Ответь на вопрос чистым текстом кратко используя контекст ниже:
{context}

Вопрос: {question}

Ответ:"""
PROMPT = PromptTemplate.from_template(TEMPLATE)


models_lock = asyncio.Lock()

filter_model = None
filter_tokenizer = None
embeddings = None
sparse_embeddings = None
rerank_model = None
llm_model = None
client = None
rerank_retriever = None
MODELS_LOADED = False
QDRANT_LOADED = False

def load_models():
    global filter_model
    global filter_tokenizer
    global embeddings
    global sparse_embeddings
    global rerank_model
    global llm_model
    global MODELS_LOADED

    if MODELS_LOADED:
        return

    _ = snapshot_download(
        repo_id=FILTER_MODEL_NAME,
        local_dir=FILTER_MODEL_DIR,
        ignore_patterns=['*.bin', 'onnx/', 'openvino/']
    )
    filter_model = AutoModelForSequenceClassification.from_pretrained(
        FILTER_MODEL_DIR,
        device_map='auto'
    )
    filter_tokenizer = AutoTokenizer.from_pretrained(FILTER_MODEL_DIR)
    filter_model.eval()
    logger.info('filter model loaded')

    _ = snapshot_download(
        repo_id=EMBEDS_MODEL_NAME,
        local_dir=EMBEDS_LOCAL_DIR,
        ignore_patterns=['*.bin', 'onnx/', 'openvino/']
    )
    query_encode_prefix = 'task: search result | query: '
    doc_encode_prefix = 'title: none | text: '
    logger.info(f'doc_encode_prefix: {doc_encode_prefix}')
    logger.info(f'query_encode_prefix: {query_encode_prefix}')
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDS_LOCAL_DIR,
        encode_kwargs={'prompt': doc_encode_prefix},
        query_encode_kwargs={'prompt': query_encode_prefix},
    )
    sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDS_NAME)
    logger.info('embeddings model loaded')

    _ = snapshot_download(
        repo_id=RERANK_MODEL_NAME,
        local_dir=RERANK_LOCAL_DIR,
        ignore_patterns=['*.bin', 'onnx/', 'openvino/']
    )
    rerank_model = AutoModel.from_pretrained(
        RERANK_LOCAL_DIR,
        dtype='auto',
        device_map='auto',
        trust_remote_code=True,
    )
    rerank_model.eval()
    logger.info(rerank_model.device)
    logger.info('rerank model loaded')

    _ = snapshot_download(
        repo_id=LLM_MODEL_REPO,
        local_dir=LLM_LOCAL_DIR,
        allow_patterns=LLM_MODEL_NAME
    )

    llm_model = Llama(
        model_path=f'{LLM_LOCAL_DIR}/{LLM_MODEL_NAME}',
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False
    )
    MODELS_LOADED = True
    logger.info('llm model loaded')


def filter_model_predict(question, filter_model, filter_tokenizer):
    inputs = filter_tokenizer(question.lower(), return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        logits = filter_model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return predicted_class_id

def rerank_documents(question, documents, rerank_model, include_scores=False, batch_size=2):
    content = [doc.page_content for doc in documents]
    results = rerank_model.rerank(question, content)
    indexes, scores = zip(*[(result['index'], result['relevance_score']) for result in results])
    documents = [documents[index] for index in indexes]
    documents = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    if not include_scores:
        documents, _ = zip(*documents)
    return list(documents)


class RerankRetriever:
    def __init__(self, retriever, rerank_model, rerank_factor=2):
        self.retriever = retriever
        self.rerank_model = rerank_model
        self.rerank_factor = rerank_factor


    def invoke(self, question, k=3, k_prefetch=None):
        documents = self.retriever.invoke(question, k * self.rerank_factor)
        documents = rerank_documents(question, documents, self.rerank_model, include_scores=False, batch_size=2)
        documents = documents[:k]
        return documents


class HybridRetriever:
    def __init__(self, client, collection_name, embeddings, sparse_embeddings, retrieval_type='hybrid'):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.sparse_embeddings = sparse_embeddings
        assert retrieval_type in ['hybrid', 'dense', 'sparse', 'all']
        self.retrieval_type = retrieval_type


    def invoke(self, question, k=3, k_prefetch=None):
        if k_prefetch is None:
            k_prefetch = k * 10
        query_sparse_embeddings = self.sparse_embeddings.embed_query(question)
        query_sparse_embeddings = models.SparseVector(
            indices=query_sparse_embeddings.indices,
            values=query_sparse_embeddings.values
        )
        query_dense_embeddings = self.embeddings.embed_query(question)
        points = []
        if self.retrieval_type == 'hybrid':
            prefetch = [
                models.Prefetch(
                    query=query_dense_embeddings,
                    using='dense',
                    limit=k * 2,
                ),
                models.Prefetch(
                    query=query_sparse_embeddings,
                    using='sparse',
                    limit=k * 2,
                )
            ]
            query_result = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                with_payload=True,
                limit=k
            )
            points += query_result.points
        if self.retrieval_type == 'dense' or self.retrieval_type == 'all':
            query_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_dense_embeddings,
                using='dense',
                limit=k
            )
            points += query_result.points
        if self.retrieval_type == 'sparse' or self.retrieval_type == 'all':
            query_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_sparse_embeddings,
                using='sparse',
                limit=k
            )
            points += query_result.points
        query_result = [point.payload for point in points]
        query_result = [Document(**doc) for doc in query_result]
        return query_result


def filter_docs(docs):
    return docs
    docs_filtered = []
    sources = set()
    for doc in docs:
        metadata = f'{doc.metadata["text"]}.{doc.metadata["row"]}'
        if metadata not in sources:
            sources.add(metadata)
            docs_filtered.append(doc)
    return docs_filtered


def retrieve(retriever, question, k=None):
    retrieved_docs = retriever.invoke(question, k=k) if k is not None else retriever.invoke(question)
    retrieved_docs = filter_docs(retrieved_docs)
    return retrieved_docs


def retrieve_batched(retriever, questions, max_concurrency=8, k=None):
    if k is not None:
        docs_batch = retriever.batch(questions, config={'max_concurrency': max_concurrency}, k=k)
    else:
        docs_batch = retriever.batch(questions, config={'max_concurrency': max_concurrency})
    docs_batch = [filter_docs(docs) for docs in docs_batch]
    return docs_batch


def load_qdrant():
    global client
    global rerank_retriever
    global QDRANT_LOADED

    if QDRANT_LOADED:
        return

    client = QdrantClient(path="qdrant")
    client.list_conn

    collection_name=f'embeddinggemma-300m_{CHUNK_SIZE}_{CHUNK_OVERLAP}'
    vector_size=len(embeddings.embed_query(''))
    if HYBRID and not collection_name.endswith(SUFFIX):
        collection_name += SUFFIX
    elif not HYBRID:
        assert not collection_name.endswith(SUFFIX)
    collection_info = [collection_name, vector_size, embeddings.model_name]
    if HYBRID:
        collection_info.append(SPARSE_EMBEDS_NAME)
    collection_info

    info = client.get_collection(collection_name)
    print(info.points_count, info.config.params.vectors)

    hybrid_retriver = HybridRetriever(client, collection_name, embeddings, sparse_embeddings, retrieval_type='dense')
    rerank_retriever = RerankRetriever(hybrid_retriver, rerank_model, rerank_factor=2)
    QDRANT_LOADED = True
    logger.info(f'qdrant loaded')


def generate_llm_answer(question, retriever, llm_model, prompt, max_tokens=128):
    def format_docs(docs):
        return '\n'.join([f'{doc.metadata["searchname"]}: {doc.page_content}' for doc in docs])

    def format_newlines(text):
        return text.replace('\n', ' ').strip()


    docs_retrieved = retrieve(retriever, question)
    ctx = format_docs(docs_retrieved)
    user_content = prompt.format(context=ctx, question=question)
    output = llm_model.create_chat_completion(
        messages=[{'role': 'user', 'content': user_content}],
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        max_tokens=max_tokens
    )
    answer = output['choices'][0]['message']['content']

    return answer


def generate_rag_answer(question):
    label = filter_model_predict(question, filter_model, filter_tokenizer)
    if label:
        answer = generate_llm_answer(question, rerank_retriever, llm_model, PROMPT, max_tokens=256)
        if 'нет информации' in answer:
            answer = 'К сожалению, пока не обладаю информацией по данному вопросу, попробуйте задать другой.'
    else:
        answer = 'К сожалению, ничем не могу помочь, моя специализация - вопросы по лекарственным препаратам.'
    return answer


# load_dotenv()
# MODELS_DIR = os.getenv('MODELS_DIR', 'models')
# executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)


async def get_rag_answer_impl(request: RagRequest):
    async with models_lock:
        try:
            answer = generate_rag_answer(request.question)
            return 200, {'answer': answer}
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f'Request failed with question {request.question}: {e}'
            )
