from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# DB 주소 (나중에 Docker/RDS 주소로 변경)
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:postgres@localhost:5433/testdb"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# DB 세션 가져오기용 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()