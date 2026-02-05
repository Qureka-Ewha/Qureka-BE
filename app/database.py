from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 1. DB 이름 결정: 팀에서 'lecture_db'를 쓰기로 했다면 아래 주소를 사용하세요.
# 만약 기존에 'testdb'에 데이터를 넣어놨다면 'testdb'로 바꾸시면 됩니다.
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:postgres@localhost:5433/lecture_db"

# 2. 엔진 생성
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# 3. 세션 설정
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. 베이스 클래스 (모든 모델의 부모)
Base = declarative_base()

# 5. DB 세션 주입 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()