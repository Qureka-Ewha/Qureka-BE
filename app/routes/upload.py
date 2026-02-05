from fastapi import APIRouter, UploadFile, File, Depends, BackgroundTasks, HTTPException, Form
from sqlalchemy.orm import Session
from ..database import get_db, SessionLocal
from .. import models
from ..services import processing
import shutil, os

router = APIRouter()

# 개발자 2의 폴더 구조 유지
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------
# [Background Task] 확정 시 실행되는 임베딩 작업
# ---------------------------------------------------------
def process_file_chunks(file_id: int):
    db = SessionLocal()
    try:
        file_rec = db.query(models.UploadedFile).filter_by(id=file_id).first()
        if not file_rec or not file_rec.text_content:
            return

        # 1. 텍스트를 의미 단위로 분할 (processing.py 엔진)
        chunks = processing.chunk_text(file_rec.text_content)
        # 2. 벡터 변환 (processing.py 엔진)
        embeddings = processing.get_embeddings(chunks)

        # 3. DB 저장 (페이지 번호 포함)
        for i, (content, emb) in enumerate(zip(chunks, embeddings)):
            new_chunk = models.LectureChunk(
                lecture_id=file_rec.lecture_id,
                file_id=file_id,
                content=content,
                embedding=emb,
                page_number=i + 1  # 순차적으로 페이지 번호 부여
            )
            db.add(new_chunk)
        
        db.commit()
    finally:
        db.close()

# ---------------------------------------------------------
# 1️⃣ [POST] 파일 업로드 및 텍스트 추출 (개발자 2 구조 + 개발자 1 엔진)
# ---------------------------------------------------------
@router.post("/upload")
async def upload_file(
    lecture_title: str = Form(...), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    # 1. 파일 물리적 저장
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. 엔진 가동 (개발자 1의 고성능 엔진 호출)
    if file.filename.lower().endswith(".pdf"):
        with open(file_path, "rb") as f:
            raw_text = processing.extract_from_pdf(f.read())
    elif file.filename.lower().endswith((".mp3", ".wav", ".m4a")):
        raw_text = processing.extract_from_audio(file_path, subject_hint=lecture_title)
    else:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    # 3. 강의(Lecture) 찾기 또는 자동 생성 (개발자 2 로직)
    lecture = db.query(models.Lecture).filter(models.Lecture.title == lecture_title).first()
    if not lecture:
        lecture = models.Lecture(title=lecture_title, user_id=1) # 임시 user_id 1
        db.add(lecture)
        db.commit()
        db.refresh(lecture)

    # 4. 업로드 파일 정보 저장 (is_confirmed=False)
    new_file = models.UploadedFile(
        lecture_id=lecture.id,
        file_url=file_path,
        file_type=file.content_type,
        text_content=raw_text,
        is_confirmed=False
    )
    db.add(new_file)
    db.commit()
    db.refresh(new_file)

    return {
        "file_id": new_file.id,
        "lecture_id": lecture.id,
        "extracted_text": raw_text
    }

# ---------------------------------------------------------
# 2️⃣ [PUT] 중간 저장 API (개발자 2 로직)
# ---------------------------------------------------------
@router.put("/upload/{file_id}/text")
async def update_text(file_id: int, text_content: str = Form(...), db: Session = Depends(get_db)):
    file_rec = db.query(models.UploadedFile).filter_by(id=file_id).first()
    if not file_rec:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    if file_rec.is_confirmed:
        raise HTTPException(status_code=400, detail="이미 확정된 파일은 수정할 수 없습니다.")

    file_rec.text_content = text_content
    db.commit()
    return {"message": "임시 저장되었습니다."}

# ---------------------------------------------------------
# 3️⃣ [POST] 최종 확정 API (임베딩 시작)
# ---------------------------------------------------------
@router.post("/confirm/{file_id}")
async def confirm_file(file_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    file_rec = db.query(models.UploadedFile).filter_by(id=file_id).first()
    if not file_rec:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    file_rec.is_confirmed = True
    db.commit()

    # 무거운 작업은 백그라운드에서 실행
    background_tasks.add_task(process_file_chunks, file_id)
    return {"message": "확정 완료! AI가 자료 학습을 시작합니다."}