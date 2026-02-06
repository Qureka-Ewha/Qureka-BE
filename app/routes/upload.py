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

        # 1. 텍스트 분할 시 이제 (내용, 페이지번호) 쌍을 받습니다.
        # 기존 extract_from_pdf가 List[Tuple]을 반환하도록 processing.py를 수정했으므로
        # 여기서는 text_content를 기반으로 하되, PDF일 경우 재추출하여 페이지 정보를 가져오는 것이 정확합니다.
        if file_rec.file_url.lower().endswith(".pdf"):
            with open(file_rec.file_url, "rb") as f:
                pages_data = processing.extract_from_pdf(f.read())
            chunks_with_page = processing.chunk_text_with_page(pages_data)
        else:
            # 오디오 등은 페이지가 없으므로 기본값 1 부여
            chunks = processing.chunk_text(file_rec.text_content)
            chunks_with_page = [(c, 1) for c in chunks]

        # 2. 벡터 변환 (텍스트 내용만 추출해서 보냄)
        texts_only = [c[0] for c in chunks_with_page]
        embeddings = processing.get_embeddings(texts_only)

        # 3. DB 저장 (실제 페이지 번호 반영)
        for (content, real_page), emb in zip(chunks_with_page, embeddings):
            new_chunk = models.LectureChunk(
                lecture_id=file_rec.lecture_id,
                file_id=file_id,
                content=content,
                embedding=emb,
                page_number=real_page  # i+1 대신 실제 추출된 페이지 번호 사용
            )
            db.add(new_chunk)
        
        db.commit()
    finally:
        db.close()

# ---------------------------------------------------------
# 1️⃣ [POST] 파일 업로드 및 텍스트 추출
# ---------------------------------------------------------
@router.post("/upload")
async def upload_file(
    lecture_title: str = Form(...), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.lower().endswith(".pdf"):
        with open(file_path, "rb") as f:
            # extract_from_pdf가 이제 List[Tuple]을 반환하므로 텍스트만 합쳐서 저장
            pages_data = processing.extract_from_pdf(f.read())
            raw_text = "\n".join([p[0] for p in pages_data])
    elif file.filename.lower().endswith((".mp3", ".wav", ".m4a")):
        raw_text = processing.extract_from_audio(file_path, subject_hint=lecture_title)
    else:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    lecture = db.query(models.Lecture).filter(models.Lecture.title == lecture_title).first()
    if not lecture:
        lecture = models.Lecture(title=lecture_title, user_id=1) 
        db.add(lecture)
        db.commit()
        db.refresh(lecture)

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
# 2️⃣ [PUT] 중간 저장 API
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
# 3️⃣ [POST] 최종 확정 API
# ---------------------------------------------------------
@router.post("/confirm/{file_id}")
async def confirm_file(file_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    file_rec = db.query(models.UploadedFile).filter_by(id=file_id).first()
    if not file_rec:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    file_rec.is_confirmed = True
    db.commit()

    background_tasks.add_task(process_file_chunks, file_id)
    return {"message": "확정 완료! AI가 자료 학습을 시작합니다."}