from abc import ABC, abstractmethod
from typing import Optional, List
from domain.entities.exam import Exam

class ExamRepository(ABC):
    @abstractmethod
    async def save(self, exam: Exam) -> None:
        ...

    @abstractmethod
    async def get(self, exam_id: str) -> Optional[Exam]:
        ...

    @abstractmethod
    async def list(self) -> List[Exam]:
        ...
