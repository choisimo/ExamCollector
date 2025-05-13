from fastapi import FastAPI
from prometheus_client import start_http_server, Summary
from fastapi import Request
from fastapi.responses import RedirectResponse
import sys
import os
# Add src directory to PYTHONPATH for absolute imports
sys.path.insert(0, os.path.dirname(__file__))
from presentation.api.exam import router as exam_router

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

app = FastAPI(title="ExamCollector API")

@app.on_event("startup")
async def startup_event():
    start_http_server(8001)

app.include_router(exam_router, prefix="/exams", tags=["Exams"])

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root URL to docs."""
    return RedirectResponse(url="/docs")

@app.middleware("http")
@REQUEST_TIME.time()
async def metrics_middleware(request: Request, call_next):
    response = await call_next(request)
    return response
