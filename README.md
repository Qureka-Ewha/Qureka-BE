# Qureka - Backend

Qureka 서비스의 백엔드 애플리케이션으로, 자기주도 학습을 하는 대학생을 위한 강의자료 기반 개인 맞춤형 소크라테스식 질문 생성 AI 튜터링 서비스입니다.
백엔드는 멀티모달 파일(PDF, 이미지, 음성)의 전처리 및 벡터화, 소크라테스식 AI 대화 제어, RAG 기반 유사도 검색, 그리고 학습 리포트 시각화를 위한 데이터 가공을 담당합니다.

- **시연 영상**: https://youtu.be/MqKK-fWk_9g

---

## 🛠 기술 스택

- **Framework**: FastAPI
- **Language**: Python
- **Database**: PostgreSQL, pgvector (Docker 기반)
- **ORM & Auth**: SQLAlchemy, python-jose (JWT)
- **AI & ML**: Gemini 2.5 Flash, Gemini-Embedding-001, OpenAI Whisper
- **Data Processing**: PyMuPDF

---

## 📐 시스템 내 백엔드 역할

Qureka 시스템은 다음과 같은 아키텍처로 구성됩니다.

- **Frontend (React)**
  사용자의 멀티모달 학습 자료 업로드 UI 제공, AI 튜터와의 실시간 채팅 인터페이스 렌더링, 핵심 개념 트리 및 이해도 타임라인 시각화를 담당합니다.
- **Backend (FastAPI)**
  - **Auth & Lecture App**: 사용자 인증(JWT) 및 강의/세션 메타데이터 관리.
  - **Pipeline App**: 업로드된 파일의 전처리(OCR/STT), 텍스트 청킹(Chunking), 임베딩 및 pgvector 저장.
  - **RAG & Chat App**: 사용자 질의 시 L2 Distance 기반 문맥 유사도 검색, 소크라테스식 튜터링 프롬프트 결합 및 생성 제어.
  - **Report App**: 대화 로그를 분석하여 프론트엔드 렌더링(D3.js)을 위한 마인드맵 및 타임라인 JSON 데이터 가공.

---

## 📂 디렉터리 구조

~~~text
📦 Qureka-BE
 ┣━ 📂 .vscode                 # VS Code 환경 설정
 ┣━ 📂 app
 ┃  ┣━ 📂 routes               # API 라우터 (엔드포인트)
 ┃  ┃  ┣━ 📜 __init__.py
 ┃  ┃  ┣━ 📜 chat.py           # 채팅 및 RAG 관련 API
 ┃  ┃  ┣━ 📜 report.py         # 학습 리포트 생성 API
 ┃  ┃  ┣━ 📜 upload.py         # 파일 전처리 및 파이프라인 API
 ┃  ┃  ┗━ 📜 users.py          # 사용자 계정 및 세션 관리 API
 ┃  ┣━ 📂 services             # 핵심 비즈니스 로직 및 연동 모듈
 ┃  ┃  ┣━ 📜 academic_concepts.py # 개념 추출 및 프롬프트 로직
 ┃  ┃  ┣━ 📜 processing.py     # 멀티모달 텍스트 추출 및 임베딩
 ┃  ┃  ┗━ 📜 report.py         # 시각화 리포트용 데이터 가공 로직
 ┃  ┣━ 📜 __init__.py
 ┃  ┣━ 📜 auth.py              # JWT 인증 및 보안 처리
 ┃  ┣━ 📜 database.py          # 데이터베이스 연결 및 세션 관리
 ┃  ┣━ 📜 main.py              # FastAPI 애플리케이션 진입점
 ┃  ┣━ 📜 models.py            # SQLAlchemy DB 테이블 모델
 ┃  ┗━ 📜 schemas.py           # Pydantic 데이터 검증 및 입출력 스키마
 ┣━ 📂 uploaded_files          # 업로드된 파일 임시 저장 경로
 ┣━ 📜 .gitignore
 ┣━ 📜 docker-compose.yml      # PostgreSQL 및 pgvector 컨테이너 실행 설정
 ┗━ 📜 requirements.txt        # 파이썬 패키지 의존성 목록
~~~

---

## 🚀 실행 방법

### 1. 실행 환경
- Python 3.9 이상
- Docker 및 Docker Compose (데이터베이스 실행용)
- 프론트엔드 연동을 위한 로컬 서버 환경 설정

### 2. 환경 변수 설정
프로젝트 최상단 디렉터리에 `.env` 파일을 생성하고 필요한 API 키와 DB 주소를 입력합니다.

~~~env
DATABASE_URL=postgresql://사용자이름:비밀번호@localhost:5432/데이터베이스이름
GOOGLE_API_KEY=발급받은_Gemini_API_KEY
JWT_SECRET_KEY=발급받은_시크릿_키
~~~

### 3. 데이터베이스 실행
Docker를 이용하여 pgvector가 포함된 PostgreSQL 컨테이너를 백그라운드에서 실행합니다.

~~~bash
docker-compose up -d
~~~

### 4. 가상환경 및 의존성 설치
~~~bash
# 가상환경 생성 및 활성화
python -m venv venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
~~~

### 5. 서버 실행
~~~bash
uvicorn app.main:app --reload
~~~

---

## 🧪 테스트 방법

자동화된 테스트 코드는 포함되어 있지 않으며, 다음 시나리오 기반으로 시스템 안정성과 AI 품질을 검증합니다.

1. **대기 시간(Latency) 테스트**: Background Tasks 비동기 큐잉을 통해 대용량 PDF 및 손글씨 업로드 시 체감 대기 시간이 1분 내외로 단축되는지 확인.
2. **시스템 연속성 검증**: LLM 연산 부하로 인한 간헐적 API 타임아웃 발생 시 서버가 다운되지 않고 디폴트 Fallback 힌트 질문이 정상 응답하는지 테스트.
3. **환각 억제율(Grounding) 검증**: pgvector의 L2 Distance 유사도 검색(Top-3 추출)을 통해 AI가 외부 정보 혼입 없이 철저히 교안 내에서만 역질문을 생성하는지 검증.

---

## 🔗 프론트엔드 연동

백엔드는 배포된(혹은 로컬의) React 프론트엔드 서버와 연동됩니다.
- **REST API 제공**: FastAPI
- **인증 방식**: JWT 토큰 (Authorization: Bearer Token) 기반 사용자 검증
- 프론트엔드 실행 방법 및 화면 구성은 프론트엔드 저장소의 README를 참고하십시오.

---

## 📚 사용한 오픈소스

- [FastAPI](https://fastapi.tiangolo.com/)
- [PostgreSQL](https://www.postgresql.org/)
- [pgvector](https://github.com/pgvector/pgvector)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [python-jose](https://github.com/mpdavis/python-jose)
