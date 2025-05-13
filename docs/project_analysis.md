# ExamCollector 프로젝트 현황 및 개선 필요사항
#### last update : 2025-05-13 


## 1. 현재 구현된 부분

1. **클린 아키텍처**
   - `presentation` / `application` / `domain` / `infrastructure` 계층 분리
   - FastAPI 앱 초기화 및 라우터 관리 (`src/main.py`)

2. **도메인 & 유스케이스**
   - `Exam` 엔티티 (`src/domain/entities/exam.py`)
   - `ExamRepository` 인터페이스 (`src/domain/repositories/exam_repo.py`)
   - `CreateExamUseCase` 구현 (`src/application/usecases/create_exam.py`)

3. **프레젠테이션 (API)**
   - CRUD: `/exams` 엔드포인트 (`src/presentation/api/exam.py`)
   - 문서 분석: `/exams/upload` 엔드포인트
   - Pydantic 스키마 (`src/presentation/api/schemas.py`)

4. **문서 분석 & 클러스터링**
   - `DocumentParserService` (PDF, DOCX, TXT, HWP) (`src/application/services/document_parser.py`)
   - `DocumentClusterService` (TF-IDF + KMeans) (`src/application/services/document_cluster.py`)

5. **플러그인 & i18n**
   - `Plugin` 베이스 및 자동 발견/관리 (`src/plugins`)
   - `LocaleManager` / locale JSON (`src/i18n`)

6. **인프라 & 모니터링**
   - In-Memory 레포지토 (`src/infrastructure/db/in_memory_repo.py`)
   - Prometheus 미들웨어 (`src/main.py`)
   - 환경설정 관리 (`src/infrastructure/config/settings.py`)

7. **OCR 옵션**
   - 로컬(Tesseract), AWS Textract, Google Vision, OCR.Space 설정
   - 관련 패키지: `pytesseract`, `boto3`, `google-cloud-vision`, `httpx`

8. **의존성 & CI/CD**
   - Poetry (`pyproject.toml`)
   - GitHub Actions (`.github/workflows/pipeline.yml`)


## 2. 개선 필요사항

1. **영속 DB 연동 & 샤딩**
   - PostgreSQL Async + Alembic
   - Sharding & JSONB 인덱스

2. **문서 레이아웃 분석**
   - LayoutParser/Detectron2로 섹션 분리
   - OCR 전처리 흐름 관리

3. **OCR 공급자 분기 로직**
   - `OCR_PROVIDER` 기반 서비스 구현
   - AWS/Google/OCR.Space API 연동 및 예외 처리

4. **HWP 파싱**
   - `hwp5` 또는 외부 서비스 연동

5. **테스트 커버리지**
   - 유닛/통합 테스트 추가 (pytest, TestClient)
   - CI 커버리지 리포트

6. **문서화 & 배포**
   - `docs/` 내 다이어그램·가이드 추가
   - Swagger/OpenAPI 활성화

7. **보안 & 시크릿 관리**
   - `cryptography.Fernet` 설정 암호화
   - AWS KMS/Vault 연동

8. **성능 최적화**
   - Redis 캐싱 (`src/infrastructure/db/cache.py`)
   - 비동기 워커(Celery)로 백그라운드 처리
