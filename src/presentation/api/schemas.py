from pydantic import BaseModel
from typing import Dict

class ExamCreate(BaseModel):
    title: str
    metadata: Dict[str, str]

class ExamRead(ExamCreate):
    id: str
