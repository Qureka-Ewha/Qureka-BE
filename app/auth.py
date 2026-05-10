# app/auth.py
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from . import models, database # DB 및 모델 참조 추가
import os

# JWT 설정 (.env 에 JWT_SECRET_KEY 설정 권장; 없으면 개발용 기본값)
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-only-insecure-change-me")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 로그아웃 시 무효화한 액세스 토큰 (재시작 시 초기화; 운영은 Redis 등 검토)
REVOKED_TOKENS: set[str] = set()


def blacklist_token(token: str) -> None:
    REVOKED_TOKENS.add(token)

# OAuth2 설정 (Swagger UI에서 로그인 버튼 활성화 용도)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# 암호화 설정 (argon2 사용)
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# ---------------------------------------------------------
# 1. 비밀번호 관련 유틸리티
# ---------------------------------------------------------
def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# ---------------------------------------------------------
# 2. 토큰 생성 및 검증
# ---------------------------------------------------------
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ---------------------------------------------------------
# 3. [핵심 추가] 현재 로그인한 사용자 가져오기
# ---------------------------------------------------------
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)):
    """
    HTTP 헤더의 Authorization: Bearer <token>을 읽어 
    유효한 사용자인지 확인하고 DB에서 유저 객체를 반환합니다.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증 정보가 유효하지 않거나 만료되었습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if token in REVOKED_TOKENS:
        raise credentials_exception

    try:
        # 토큰 복호화
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub") # 보통 sub에 이메일을 넣습니다.
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # DB에서 사용자 확인
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise credentials_exception
        
    return user