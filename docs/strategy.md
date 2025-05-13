# ExamCollector 프로젝트 종합 분석 및 발전 방향 연구

## 서론  
ExamCollector 프로젝트는 시험 문제 수집 및 관리 시스템으로서 클린 아키텍처 원칙에 기반한 현대적 소프트웨어 설계 패턴을 채택하고 있습니다. 본 보고서는 현재 구현 현황을 다각도로 평가하고, 개선이 필요한 핵심 영역에 대한 체계적인 분석을 수행합니다. 특히 문서 처리 파이프라인의 최적화, 분산 시스템 아키텍처 강화, 생산성 향상을 위한 개발자 경험 개선 등 세 가지 주요 축을 중심으로 기술적 접근 방식을 심층적으로 탐구합니다. 연구 방법론으로는 실제 구현 코드 베이스의 정성적 평가와 아키텍처 설계 문서의 비교 분석을 결합하였으며, 이를 통해 도출된 인사이트를 체계화하였습니다.

## 프로젝트 현황 평가

### 아키텍처 구현 수준
4계층 클린 아키텍처가 `presentation`, `application`, `domain`, `infrastructure`로 명확하게 분리되어 있으며, FastAPI 기반의 API 게이트웨이 패턴이 효과적으로 적용되었습니다. `src/main.py`에서의 앱 초기화 로직은 의존성 주입 프레임워크를 활용한 모듈식 구성이 두드러집니다. Prometheus 미들웨어 통합을 통해 모니터링 기능이 기본 제공되는 점은 프로덕션 환경 대비성을 높이는 중요한 요소로 평가됩니다.

도메인 주도 설계(DDD) 원칙이 `Exam` 엔티티와 `ExamRepository` 인터페이스에 반영되어 있으나, 애그리게이트 루트와 값 객체(Value Object) 패턴의 적용 미비로 인해 복잡한 비즈니스 로직 처리에 한계가 존재합니다. `CreateExamUseCase` 구현체에서의 트랜잭션 관리 전략은 향후 영속성 계층과의 통합 시 재검토가 필요한 부분입니다.

### 문서 처리 시스템
다양한 파일 형식(PDF, DOCX, TXT, HWP)을 수용하는 `DocumentParserService`와 TF-IDF 기반의 `DocumentClusterService`가 구현되어 기본적인 문서 분류 기능을 수행합니다. 그러나 레이아웃 분석 기능의 부재로 인해 표 또는 수식이 포함된 문서 처리 시 정확도 저하 문제가 예상됩니다. HWP 파싱을 위한 `hwp5` 라이브러리 통합은 현재 프로토타입 수준에 머물러 있어 실제 운영 환경 적용 전 추가 개발이 필요합니다.

OCR 기능은 Tesseract, AWS Textract, Google Vision, OCR.Space 등 주요 제공업체를 지원하지만, 공급자별 API 호환성 레이어가 미비하여 전환 비용이 높은 구조적 문제점을 안고 있습니다. 특히 비동기 처리 메커니즘이 완전히 구현되지 않아 대용량 문서 처리 시 성능 병목 현상이 발생할 가능성이 있습니다.

### 확장성 및 유지보수성
플러그인 시스템의 자동 발견 메커니즘과 `LocaleManager`를 통한 다국어 지원은 시스템의 확장성을 크게 향상시키는 요소입니다. 그러나 현재 인프라 계층의 In-Memory 저장소 구현체는 개발 단계의 편의성을 위해 채택된 임시 방편에 불과하며, 실제 서비스 운영을 위해서는 분산 트랜잭션을 지원하는 영속성 계층으로의 전환이 필수적입니다.

CI/CD 파이프라인은 GitHub Actions을 통해 기본적인 빌드 및 테스트 작업을 수행하지만, E2E 테스트 케이스와 카나리아 배포 전략의 부재로 인해 지속적 전달(Continuous Delivery) 측면에서 미흡한 부분이 있습니다. Poetry를 통한 의존성 관리는 패키지 버전 충돌 위험을 최소화하는 데 기여하고 있습니다.

## 핵심 개선 과제 분석

### 영속성 계층 고도화
PostgreSQL Async 드라이버와 Alembic 마이그레이션 도구의 통합은 ACID 트랜잭션 보장과 스키마 버전 관리를 가능하게 합니다. 샤딩 전략 수립 시 문서의 지리적 분포 특성을 고려한 동적 샤드 키 할당 알고리즘의 도입이 필요하며, JSONB 인덱싱을 통해 반정형 데이터 처리 효율을 극대화할 수 있습니다. 분산 트랜잭션 관리를 위한 Saga 패턴 구현체의 개발이 병행되어야 합니다.

```python
# 샤드 키 생성 예시 (지리적 해시 기반)
import hashlib

def generate_shard_key(latitude: float, longitude: float) -> str:
    geo_hash = hashlib.sha256(f"{latitude}:{longitude}".encode()).hexdigest()
    return geo_hash[:8]
```

### 문서 레이아웃 인식 강화
Detectron2 기반의 LayoutParser 도입을 통해 문서 내 표, 그림, 수식 영역을 정확하게 식별할 수 있습니다. OCR 전처리 파이프라인에는 이미지 보정(dewarping), 노이즈 제거, 해상도 최적화 단계를 추가하여 인식률을 개선해야 합니다. 특히 한글 문서 특성에 맞춘 커스텀 학습 모델 개발이 중요하며, Transfer Learning 기법을 활용한 도메인 적응 전략이 효과적일 것입니다.

### OCR 서비스 통합 아키텍처
공급자별 API 추상화 계층을 설계하여 `OCR_PROVIDER` 환경 변수에 따른 동적 라우팅 메커니즘을 구현해야 합니다. Circuit Breaker 패턴을 적용한 장애 조치(failover) 시스템과 요청 비용 최적화를 위한 지능형 라우팅 알고리즘의 개발이 필요합니다. 각 제공업체별 Rate Limit 정책을 고려한 배치 처리 큐 시스템의 도입으로 처리량을 극대화할 수 있습니다.

```python
# 추상화 계층 예시
from abc import ABC, abstractmethod
import boto3

class OCRProvider(ABC):
    @abstractmethod
    async def recognize(self, image: bytes) -> str:
        pass

class AWSOCR(OCRProvider):
    def __init__(self):
        self.client = boto3.client('textract')

    async def recognize(self, image: bytes) -> str:
        response = self.client.detect_document_text(Document={'Bytes': image})
        return " ".join([item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE'])
```

### 테스트 자동화 전략
pytest와 FastAPI TestClient를 결합한 E2E 테스트 프레임워크 구축이 시급합니다. 모의 객체(Mock)를 활용한 서비스 간 결합도 테스트와 Property-Based Testing 기법의 도입으로 엣지 케이스 커버리지를 확장해야 합니다. CI 파이프라인에 Codecov 연동을 통해 테스트 커버리지 리포트를 자동화하고, mutation testing 도구를 활용한 테스트 품질 검증 체계를 마련해야 합니다.

## 시스템 최적화 방안

### 캐싱 메커니즘 구현
Redis를 이용한 L1/L2 캐시 계층화 전략을 수립하고, LRU(Least Recently Used) 캐시 교체 알고리즘을 적용하여 히트율을 최적화합니다. 문서 메타데이터는 인메모리 캐시에, 대용량 콘텐츠는 디스크 기반 캐시에 저장하는 계층적 접근 방식이 효과적입니다. Cache-Aside 패턴과 Write-Behind 패턴을 상황에 따라 선택 적용하는 유연한 캐싱 정책이 필요합니다.

### 비동기 작업 처리
Celery 워커와 Redis 브로커를 연동한 분산 태스크 큐 시스템을 구축합니다. CPU 집약적 OCR 처리는 전용 워커 풀에서 실행되도록 구성하고, 우선순위 큐를 통해 긴급 작업의 신속한 처리를 보장해야 합니다. 작업 상태 추적을 위한 Flower 모니터링 대시보드의 통합이 필수적이며, 재시도 정책과 데드 레터 큐(Dead Letter Queue) 구현으로 작업 손실을 방지합니다。

```python
# Celery 태스크 예시
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task(queue='high_priority')
def process_ocr(document_id: str):
    document = get_document(document_id)
    result = ocr_provider.recognize(document.content)
    save_ocr_result(document_id, result)
```

### 보안 강화 조치
Fernet 키를 이용한 설정 파일 암호화와 AWS KMS 기반의 Secrets Manager 연동으로 민감정보 보호 수준을 강화합니다. API 게이트웨이 수준에서 JWT 토큰 검증 미들웨어를 추가하고, OAuth 2.0 인증 플로우를 지원하는 플러그인 개발이 필요합니다. 정적 코드 분석 도구(SonarQube)와 동적 분석 도구(OWASP ZAP)를 CI 파이프라인에 통합하여 취약점을 사전에 차단해야 합니다.

## 결론 및 향후 계획
ExamCollector 프로젝트는 현대적 소프트웨어 엔지니어링 원칙을 충실히 반영한 우수한 기반 구조를 갖추고 있으나, 프로덕션 환경 적용을 위해서는 문서 처리 파이프라인의 고도화와 분산 시스템 아키텍처 강화가 시급합니다. 개선 작업의 우선순위를 영속성 계층 구축(1단계), 테스트 자동화(2단계), 보안 강화(3단계)로 구분하여 단계적 접근이 바람직합니다. 향후 연구 과제로는 AI 기반 문서 분류 모델의 통합과 크로스플랫폼 클라이언트 SDK 개발을 제안하며, 오픈소스 생태계 참여를 통한 기술 표준화 노력이 필요할 것입니다. 
