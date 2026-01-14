from pydantic import BaseModel, EmailStr
from typing import Optional

# 회원가입 시 받을 데이터
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    nickname: str
    department: str
    grade: int

# 로그인 성공 시 돌려줄 데이터
class Token(BaseModel):
    access_token: str
    token_type: str