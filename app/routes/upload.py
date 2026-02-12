from fastapi import APIRouter, UploadFile, File, Depends, BackgroundTasks, HTTPException, Form
from sqlalchemy.orm import Session
from ..database import get_db, SessionLocal
from .. import models
from ..services import processing
import shutil, os

router = APIRouter()

# 업로드 경로 설정
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

        # PDF일 경우 시각적 분석 기반으로 재추출하여 페이지 정보 확보
        if file_rec.file_url.lower().endswith(".pdf"):
            with open(file_rec.file_url, "rb") as f:
                # [중요] processing.py에서 수정한 extract_from_pdf 호출
                pages_data = processing.extract_from_pdf(f.read())
            chunks_with_page = processing.chunk_text_with_page(pages_data)
        else:
            # 오디오 등은 페이지가 없으므로 기본값 1 부여
            # (processing.py에 chunk_text가 없다면 chunk_text_with_page를 활용하도록 수정 가능)
            chunks_with_page = [(file_rec.text_content, 1)]

        # 벡터 변환
        texts_only = [c[0] for c in chunks_with_page]
        embeddings = processing.get_embeddings(texts_only)

        # DB 저장
        for (content, real_page), emb in zip(chunks_with_page, embeddings):
            new_chunk = models.LectureChunk(
                lecture_id=file_rec.lecture_id,
                file_id=file_id,
                content=content,
                embedding=emb,
                page_number=real_page
            )
            db.add(new_chunk)
        
        db.commit()
    finally:
        db.close()

# ---------------------------------------------------------
# 1️⃣ [POST] 파일 업로드 및 텍스트 추출 (Gemini Vision 강제)
# ---------------------------------------------------------
@router.post("/upload")
async def upload_file(
    lecture_title: str = Form(...), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    # 파일 저장
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 텍스트 추출 로직
    if file.filename.lower().endswith(".pdf"):
        print(f"📡 PDF 업로드 감지: Gemini Vision 분석 시작 - {file.filename}")
        with open(file_path, "rb") as f:
            # [핵심] processing.py의 Vision 전용 함수 호출
            pages_data = processing.extract_from_pdf(f.read())
            # 손글씨가 포함된 전체 텍스트 합치기
            raw_text = "\n".join([p[0] for p in pages_data])
    elif file.filename.lower().endswith((".mp3", ".wav", ".m4a")):
        raw_text = processing.transcribe_audio(file_path) # 함수명 통일 확인 필요
    else:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    # 강의(Lecture) 레코드 확인 및 생성
    lecture = db.query(models.Lecture).filter(models.Lecture.title == lecture_title).first()
    if not lecture:
        lecture = models.Lecture(title=lecture_title, user_id=1) 
        db.add(lecture)
        db.commit()
        db.refresh(lecture)

    # 업로드 파일 레코드 생성
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

# (이하 update_text, confirm_file API는 기존과 동일하여 생략 가능하지만 구조 유지를 위해 포함)
@router.put("/upload/{file_id}/text")
async def update_text(file_id: int, text_content: str = Form(...), db: Session = Depends(get_db)):
    file_rec = db.query(models.UploadedFile).filter_by(id=file_id).first()
    if not file_rec: raise HTTPException(status_code=404)
    file_rec.text_content = text_content
    db.commit()
    return {"message": "임시 저장되었습니다."}

@router.post("/confirm/{file_id}")
async def confirm_file(file_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    file_rec = db.query(models.UploadedFile).filter_by(id=file_id).first()
    if not file_rec: raise HTTPException(status_code=404)
    file_rec.is_confirmed = True
    db.commit()
    background_tasks.add_task(process_file_chunks, file_id)
    return {"message": "확정 완료!"}