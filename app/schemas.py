from pydantic import BaseModel, EmailStr

# 회원가입 시 받을 데이터 양식
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    nickname: str
    department: str
    grade: int

# 로그인 시 받을 데이터 양식 (이게 빠져서 에러가 난 거예요!)
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# 로그인이 성공했을 때 응답할 토큰 양식
class Token(BaseModel):
    access_token: str
    token_type: str