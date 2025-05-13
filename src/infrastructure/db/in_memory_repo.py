from typing import Dict, List, Optional
from domain.repositories.exam_repo import ExamRepository
from domain.entities.exam import Exam

class InMemoryExamRepository(ExamRepository):
    def __init__(self):
        self._store: Dict[str, Exam] = {}

    async def save(self, exam: Exam) -> None:
        self._store[exam.id] = exam

    async def get(self, exam_id: str) -> Optional[Exam]:
        return self._store.get(exam_id)

    async def list(self) -> List[Exam]:
        return list(self._store.values())
