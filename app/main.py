# app/main.py
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from . import models, schemas, auth, database
from .services.pdf_processing import extract_text_from_pdf
import shutil
import os
from .services.audio_processing import transcribe_long_audio

app = FastAPI()

# 서버 시작 시 테이블 자동 생성 (1주 차 개발 단계에서 유용)
models.Base.metadata.create_all(bind=database.engine)

@app.get("/")
def root():
    return {"message": "Qureka Auth System is Running!"}

# 회원가입 API
@app.post("/signup")
def signup(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="이미 등록된 이메일입니다.")
    
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
    return {"message": "회원가입이 완료되었습니다."}

# 로그인 API
@app.post("/login", response_model=schemas.Token)
def login(form_data: schemas.UserLogin, db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.email).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 틀렸습니다.")
    
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

# 1. 블랙리스트 저장소 (서버가 켜져 있는 동안만 유지되는 임시 저장소)
token_blacklist = set()

# 2. 로그아웃 API
@app.post("/logout")
def logout(token: str = Depends(auth.oauth2_scheme)):
    """
    리액트에서 로그아웃 버튼을 누르면 이 API를 호출합니다.
    서버는 받은 토큰을 블랙리스트에 넣어 이후 사용을 막습니다.
    """
    token_blacklist.add(token)
    return {"message": "Successfully logged out"}


# 파일 업로드 및 텍스트 추출 API
@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. 업로드된 파일의 내용을 읽습니다.
    content = await file.read()
    
    # 2. 아까 만든 함수로 텍스트를 뽑습니다.
    extracted_text = extract_text_from_pdf(content)
    
    # 3. 결과를 JSON으로 돌려줍니다.
    return {
        "filename": file.filename,
        "text": extracted_text
    }

@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    # 1. 임시 저장 폴더 확인
    temp_dir = "app/temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    # 2. 업로드된 파일 저장
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 3. 파일명에서 힌트 추출 (확장자 제거)
    subject_hint = os.path.splitext(file.filename)[0]
    
    # 4. 긴 음성 처리 함수 실행 (힌트 포함)
    extracted_text = transcribe_long_audio(file_path, subject_hint)
    
    return {
        "filename": file.filename,
        "subject_hint": subject_hint,
        "text": extracted_text
    }