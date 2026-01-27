from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False) #로그인
    hashed_password = Column(String, nullable=False) #비밀번호
    nickname = Column(String, nullable=False) # 사용자 이름
    department = Column(String)  # 과
    grade = Column(Integer)      # 학년 


class AudioResult(Base):
    __tablename__ = "audio_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    subject_hint = Column(String(255), nullable=True)
    
    # AI가 처음 변환한 텍스트
    original_text = Column(Text) 
    
    # 사용자가 수정한 최종 텍스트
    final_text = Column(Text) 
    
    # 데이터 생성 시간
    created_at = Column(DateTime(timezone=True), server_default=func.now())