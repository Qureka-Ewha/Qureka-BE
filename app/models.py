from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from pgvector.sqlalchemy import Vector  # RAG 기능을 위해 필요
from .database import Base


class User(Base):
    __tablename__ = "users" # DB에 저장될 실제 테이블 이름입니다.

    id = Column(Integer, primary_key=True, index=True) # 고유 번호(PK)입니다. index=True는 검색 속도를 높여줍니다.
    email = Column(String, unique=True, index=True, nullable=False) # 로그인 아이디입니다. 중복 안 되고 비어있으면 안 됩니다.
    hashed_password = Column(String, nullable=False) # 보안을 위해 암호화된 비밀번호를 저장합니다.
    nickname = Column(String, nullable=False) # 사용자 별명입니다.
    department = Column(String, nullable=True) # 학과 정보입니다. (개발자 1 추가분)
    grade = Column(Integer, nullable=True) # 학년 정보입니다. (개발자 1 추가분)
    created_at = Column(DateTime, default=datetime.utcnow) # 가입 일시를 자동으로 기록합니다.

    # 이 사용자가 만든 모든 강의 목록을 찾아올 수 있게 연결합니다.
    lectures = relationship("Lecture", back_populates="user")
    
class Lecture(Base):
    __tablename__ = "lectures"

    id = Column(Integer, primary_key=True)
    # 어떤 사용자의 강의인지 기록합니다. 사용자가 탈퇴하면(CASCADE) 강의도 같이 지워집니다.
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    title = Column(String, nullable=False) # 강의 제목입니다.
    created_at = Column(DateTime, default=datetime.utcnow)

    # 연결된 관계들 (이 강의에 속한 파일들, 텍스트 조각들, 채팅방들을 한 번에 부를 수 있습니다.)
    user = relationship("User", back_populates="lectures")
    files = relationship("UploadedFile", back_populates="lecture") # 이 강의에 포함된 모든 원본 파일(PDF, MP3 등)을 연결합니다.
    chunks = relationship("LectureChunk", back_populates="lecture") # 파일들로부터 쪼개진 **'AI용 텍스트 조각들'**을 연결합니다.
    chat_sessions = relationship("ChatSession", back_populates="lecture") # 이 강의와 관련해서 나눈 대화방들을 연결합니다.
 
class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True)
    lecture_id = Column(Integer, ForeignKey("lectures.id", ondelete="CASCADE"))

    file_url = Column(String, nullable=False) # 서버 어디에 파일이 저장되어 있는지 경로를 적습니다.
    file_type = Column(String, nullable=False) # PDF인지 AUDIO인지 구분합니다.
    
    # 추출 직후엔 '추출본'이 저장되고, 사용자 수정 후엔 '수정본'으로 덮어씌워집니다.
    text_content = Column(Text, nullable=True) 
    is_confirmed = Column(Boolean, default=False) # True가 되어야만 RAG 학습(Embedding) 대상이 됨
    #AI 추출 직후: text_content에 추출된 텍스트를 일단 저장합니다. (이때 is_confirmed = False)
    #프론트엔드: 사용자가 text_content를 불러와서 수정합니다.
    #검토 완료: 사용자가 수정한 내용을 다시 text_content에 덮어쓰고 is_confirmed = True로 바꿉니다.

    lecture = relationship("Lecture", back_populates="files")
    chunks = relationship("LectureChunk", back_populates="file")

    created_at = Column(DateTime, default=datetime.utcnow)
    # 파일 수정 시 자동으로 시간을 업데이트합니다.
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
 
class LectureChunk(Base):
    __tablename__ = "lecture_chunks"

    id = Column(Integer, primary_key=True)

    # 1. 소속 정보 (어떤 강의의, 어떤 파일에서 나온 조각인가?)
    lecture_id = Column(Integer, ForeignKey("lectures.id", ondelete="CASCADE"))
    file_id = Column(Integer, ForeignKey("uploaded_files.id", ondelete="CASCADE"))

    # 2. 실제 데이터 (무슨 내용이며, AI가 어떻게 이해하는가?)
    content = Column(Text, nullable=False) # 실제 텍스트 내용 (약 300~500자)
    embedding = Column(Vector(1536))       # 이 텍스트를 숫자로 변환한 '의미 값' (RAG 핵심)

    # 3. 출처 정보 (사용자에게 "어디서 찾았는지" 알려주기 위함)
    page_number = Column(Integer, nullable=True) # PDF일 경우 몇 페이지인지 기록

    # 4. 연결 통로
    lecture = relationship("Lecture", back_populates="chunks")
    file = relationship("UploadedFile", back_populates="chunks")

    created_at = Column(DateTime, default=datetime.utcnow)
 
class ChatSession(Base): # 대화방 하나하나의 정보
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True)
    lecture_id = Column(Integer, ForeignKey("lectures.id", ondelete="CASCADE"))
    title = Column(String, nullable=False) # 대화방 목록에 뜰 제목입니다.
    created_at = Column(DateTime, default=datetime.utcnow)

    lecture = relationship("Lecture", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base): # 대화방 안의 말 한마디 한마디
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role = Column(String, nullable=False) # "user"(나) 인지 "assistant"(AI) 인지 구분합니다.
    content = Column(Text, nullable=False) # 실제 대화 내용입니다.
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")