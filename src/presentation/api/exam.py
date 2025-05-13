from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Query
from typing import List, Dict

from application.usecases.create_exam import CreateExamUseCase
from domain.repositories.exam_repo import ExamRepository
from infrastructure.db.in_memory_repo import InMemoryExamRepository
from presentation.api.schemas import ExamCreate, ExamRead
from application.services.document_parser import DocumentParserService
from application.services.document_cluster import DocumentClusterService
from application.services.question_tagger import QuestionTaggerService

router = APIRouter()

# Dependency injector for repository
_repo = InMemoryExamRepository()
def get_repo() -> ExamRepository:
    """
    Returns the exam repository instance.
    
    Returns:
        ExamRepository: A dependency that provides access to the exam repository.
    """
    return _repo

def get_create_usecase(repo: ExamRepository = Depends(get_repo)) -> CreateExamUseCase:
    return CreateExamUseCase(repo)

@router.post("/", response_model=ExamRead, status_code=status.HTTP_201_CREATED)
async def create_exam(
    payload: ExamCreate,
    usecase: CreateExamUseCase = Depends(get_create_usecase),
):
    exam = await usecase.execute(payload.dict())
    return exam

@router.get("/", response_model=List[ExamRead])
async def list_exams(repo: ExamRepository = Depends(get_repo)):
    return await repo.list()

@router.get("/{exam_id}", response_model=ExamRead)
async def get_exam(exam_id: str, repo: ExamRepository = Depends(get_repo)):
    exam = await repo.get(exam_id)
    if not exam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam not found")
    return exam

@router.post("/upload", response_model=Dict[int, List[str]])
async def upload_and_cluster(
    file: UploadFile = File(...),
    parser: DocumentParserService = Depends(),
    clusterer: DocumentClusterService = Depends(lambda: DocumentClusterService(n_clusters=5)),
):
    try:
        text = await parser.parse(file)
        clusters = clusterer.cluster(text)
        return clusters
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/upload/tag", response_model=Dict[int, Dict[str, List[str]]])
async def upload_and_tag(
    file: UploadFile = File(...),
    parser: DocumentParserService = Depends(),
    clusterer: DocumentClusterService = Depends(lambda: DocumentClusterService(n_clusters=5)),
    tagger: QuestionTaggerService = Depends(),
    agent: str = Query("default", description="AI agent config name"),
    retry: bool = Query(False, description="Retry tagging if results inaccurate"),
):
    try:
        text = await parser.parse(file)
        clusters = clusterer.cluster(text)
        tags = await tagger.tag(clusters, agent_name=agent, retry=retry)
        return tags
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
