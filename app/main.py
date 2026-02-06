import os
import openai
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# 우리가 정리한 파일들 임포트
from . import models, schemas, auth, database
from .routes import upload  

# 1. 환경 설정
load_dotenv()

app = FastAPI(title="Qureka Unified Server")

# 2. 서버 실행 시 DB 테이블 자동 생성
models.Base.metadata.create_all(bind=database.engine)

# 3. 로그아웃을 위한 블랙리스트 저장소 (개발자 1 로직)
# 서버가 켜져 있는 동안 메모리에 토큰을 저장하여 무효화합니다.
token_blacklist = set()

# 4. 라우터 연결
app.include_router(upload.router, prefix="/files", tags=["Lecture Files"])

# --- 인증 관련 API (개발자 1 코드 풀버전 통합) ---

@app.post("/signup")
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

@app.post("/login", response_model=schemas.Token)
def login(form_data: schemas.UserLogin, db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.email).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="로그인 정보가 올바르지 않습니다.")
    
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

# 추가된 로그아웃 API (개발자 1 로직)
@app.post("/logout")
def logout(token: str = Depends(auth.oauth2_scheme)):
    """
    사용자가 보낸 토큰을 블랙리스트에 넣어, 이 토큰으로 더 이상 API를 호출하지 못하게 차단합니다.
    """
    token_blacklist.add(token)
    return {"message": "Successfully logged out"}

@app.get("/")
def root():
    return {"status": "running", "message": "Qureka API 통합 서버가 가동 중입니다."}