import uuid
from domain.entities.exam import Exam
from domain.repositories.exam_repo import ExamRepository

class CreateExamUseCase:
    def __init__(self, repo: ExamRepository):
        self.repo = repo

    async def execute(self, payload: dict) -> Exam:
        exam = Exam(id=str(uuid.uuid4()), **payload)
        await self.repo.save(exam)
        return exam
