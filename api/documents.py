import os
import logging
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from tools.rag import ingest_pdf, get_session_doc_stats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload/{session_id}")
async def upload_document(
    session_id: str,
    file: UploadFile = File(...),
    company: str = Form(default=""),
    doc_type: str = Form(default="annual_report"),
    year: str = Form(default=""),
):
    """Upload a PDF for a specific research session."""

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        metadata = {"doc_type": doc_type}
        if company:
            metadata["company"] = company
        if year:
            metadata["year"] = year

        result = ingest_pdf(session_id, tmp_path, metadata=metadata)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        logger.info(f"Session {session_id}: uploaded {file.filename}, {result['chunks']} chunks")
        return {
            "message": f"Ingested {file.filename}",
            "chunks": result["chunks"],
            "total_in_session": result["total_in_session"],
        }

    finally:
        os.unlink(tmp_path)


@router.get("/stats/{session_id}")
async def document_stats(session_id: str):
    """Get doc stats for a session."""
    return get_session_doc_stats(session_id)
