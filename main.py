import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router
from api.documents import router as documents_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

app = FastAPI(title="Financial Deep Research Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(documents_router)


@app.get("/")
async def root():
    return {"service": "Financial Deep Research Agent", "endpoints": {"submit_query": "POST /research/"}}


@app.get("/health")
async def health():
    return {"status": "healthy"}
