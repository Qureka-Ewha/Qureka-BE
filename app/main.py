# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import models, schemas, auth, database

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