import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import text
from dotenv import load_dotenv


# 내부 모듈 임포트
from . import models, schemas, auth, database
from .routes import upload, chat, report, users   # report, users 추가
  
# 1. 환경 설정 로드
load_dotenv()


def _upgrade_schema_postgres():
    url = str(database.engine.url)
    if "postgresql" not in url:
        return
    stmts = [
        "ALTER TABLE uploaded_files ADD COLUMN IF NOT EXISTS upload_group_id VARCHAR(64)",
        "ALTER TABLE uploaded_files ADD COLUMN IF NOT EXISTS transcript_raw TEXT",
        "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS source_kind VARCHAR(20)",
        "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS upload_group_id VARCHAR(64)",
    ]
    with database.engine.begin() as conn:
        conn.execute(text("SET lock_timeout = '2000ms'"))
        conn.execute(text("SET statement_timeout = '5000ms'"))
        for s in stmts:
            try:
                conn.execute(text(s))
            except Exception as e:
                print(f"[schema] {s}: {e}")
        try:
            conn.execute(
                text(
                    "ALTER TABLE chat_sessions DROP CONSTRAINT IF EXISTS "
                    "chat_sessions_file_id_fkey"
                )
            )
        except Exception as e:
            print(f"[schema] drop file_id fk: {e}")
        try:
            conn.execute(
                text("ALTER TABLE chat_sessions ALTER COLUMN file_id DROP NOT NULL")
            )
        except Exception as e:
            print(f"[schema] file_id nullable: {e}")
        try:
            conn.execute(
                text(
                    "ALTER TABLE chat_sessions ADD CONSTRAINT chat_sessions_file_id_fkey "
                    "FOREIGN KEY (file_id) REFERENCES uploaded_files(id) ON DELETE SET NULL"
                )
            )
        except Exception as e:
            print(f"[schema] add file_id fk: {e}")
        try:
            conn.execute(
                text(
                    "ALTER TABLE chat_messages DROP CONSTRAINT IF EXISTS "
                    "chat_messages_session_id_fkey"
                )
            )
        except Exception as e:
            print(f"[schema] drop message session fk: {e}")
        try:
            conn.execute(
                text(
                    "ALTER TABLE chat_messages ADD CONSTRAINT chat_messages_session_id_fkey "
                    "FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE"
                )
            )
        except Exception as e:
            print(f"[schema] add message session fk: {e}")

app = FastAPI(title="Qureka Unified Server")
app.mount(
    "/uploaded_files",
    StaticFiles(directory="uploaded_files"),
    name="uploaded_files"
)

# CORS: allow_credentials=True일 때 브라우저는 Allow-Origin: * 를 허용하지 않습니다.
# (크리덴셜 요청 시 API가 브라우저에서 계속 차단되는 흔한 원인)
_cors_origins_env = os.getenv("CORS_ORIGINS")
_default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
]
if _cors_origins_env and _cors_origins_env.strip():
    _allow_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    if not _allow_origins:
        _allow_origins = list(_default_origins)
else:
    _allow_origins = list(_default_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],  # OPTIONS, POST, GET 등 모든 통신 방식 허용
    allow_headers=["*"],
)

# 2. 서버 실행 시 DB 준비 (확장 설치 및 테이블 생성)
@app.on_event("startup")
def startup_event():
    """
    서버가 시작될 때 데이터베이스에 필요한 기능을 활성화하고 테이블을 생성합니다.
    """
    with database.engine.connect() as conn:
        conn.execute(text("SET statement_timeout = '5000ms'"))
        # 1) pgvector 확장 설치
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # 2) 테이블 자동 생성
    models.Base.metadata.create_all(bind=database.engine)

    # 3) 기존 PostgreSQL DB에 컬럼·FK 변경 반영 (create_all은 컬럼 추가를 안 함)
    _upgrade_schema_postgres()


# 3. 라우터 연결
app.include_router(upload.router, prefix="/files", tags=["Lecture Files"])
app.include_router(chat.router, prefix="/chat", tags=["Tutoring Chat"])
app.include_router(report.router, prefix="/analysis", tags=["Learning Report"])   # 추가
app.include_router(users.router, prefix="/users", tags=["Users"]) # Users(마이페이지) 추가!


# --- 인증 관련 API ---

@app.post("/signup", tags=["Auth"])
def signup(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()

    if db_user:
        raise HTTPException(status_code=400, detail="이미 가입된 이메일입니다.")

    hashed_pwd = auth.get_password_hash(user.password)

    new_user = models.User(
        email=user.email,
        hashed_password=hashed_pwd,
        nickname=user.nickname,
        department=user.department,
        grade=user.grade
    )

    db.add(new_user)
    db.commit()

    return {"message": "회원가입 성공"}


@app.post("/login", response_model=schemas.Token, tags=["Auth"])
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(database.get_db)
):

    user = db.query(models.User).filter(
        models.User.email == form_data.username
    ).first()

    if not user or not auth.verify_password(
            form_data.password,
            user.hashed_password
    ):
        raise HTTPException(
            status_code=401,
            detail="로그인 정보가 올바르지 않습니다."
        )

    access_token = auth.create_access_token(
        data={"sub": user.email, "user_id": user.id, "name": user.nickname} # 토큰에 유저 아이디, 이름(닉네임) 추가
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@app.post("/logout", tags=["Auth"])
def logout(token: str = Depends(auth.oauth2_scheme)):
    auth.blacklist_token(token)
    return {"message": "Successfully logged out"}


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "running",
        "message": "Qureka API 통합 서버가 가동 중입니다."
    }