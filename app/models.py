from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from pgvector.sqlalchemy import Vector  # RAG 기능을 위해 필요
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    nickname = Column(String, nullable=False)
    department = Column(String, nullable=True)
    grade = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    lectures = relationship("Lecture", back_populates="user", cascade="all, delete-orphan")

class Lecture(Base):
    __tablename__ = "lectures"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="lectures")
    files = relationship("UploadedFile", back_populates="lecture", cascade="all, delete-orphan")
    chunks = relationship("LectureChunk", back_populates="lecture", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="lecture", cascade="all, delete-orphan")

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True)
    lecture_id = Column(Integer, ForeignKey("lectures.id", ondelete="CASCADE"))
    file_url = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    text_content = Column(Text, nullable=True) 
    page_text = Column(Text, nullable=True)
    is_confirmed = Column(Boolean, default=False)

    lecture = relationship("Lecture", back_populates="files")
    chunks = relationship("LectureChunk", back_populates="file", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", cascade="all, delete-orphan")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LectureChunk(Base):
    __tablename__ = "lecture_chunks"

    id = Column(Integer, primary_key=True)
    lecture_id = Column(Integer, ForeignKey("lectures.id", ondelete="CASCADE"))
    file_id = Column(Integer, ForeignKey("uploaded_files.id", ondelete="CASCADE"))

    content = Column(Text, nullable=False)
  
    embedding = Column(Vector(3072)) 

    page_number = Column(Integer, nullable=True)

    lecture = relationship("Lecture", back_populates="chunks")
    file = relationship("UploadedFile", back_populates="chunks")

    created_at = Column(DateTime, default=datetime.utcnow)

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True)
    lecture_id = Column(Integer, ForeignKey("lectures.id", ondelete="CASCADE"))
    file_id = Column(Integer, ForeignKey("uploaded_files.id", ondelete="CASCADE"))
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    lecture = relationship("Lecture", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role = Column(String, nullable=False) # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")