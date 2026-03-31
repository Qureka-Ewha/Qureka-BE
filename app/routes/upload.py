from fastapi import APIRouter, UploadFile, File, Depends, BackgroundTasks, HTTPException, Form
from sqlalchemy.orm import Session
from ..database import get_db, SessionLocal
from .. import models
from ..services import processing
import shutil, os
import json

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
                pages_data = processing.extract_from_pdf(f.read())
            chunks_with_page = processing.chunk_text_with_page(pages_data)
        else:
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
# 1️⃣ [POST] 파일 업로드 및 텍스트 추출
# ---------------------------------------------------------
@router.post("/upload")
async def upload_file(
    lecture_title: str = Form(...), 
    user_id: int = Form(...), # ✅ 테스트를 위해 프론트에서 user_id를 받도록 추가
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    # 파일 저장 경로 설정
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 텍스트 추출 로직 (Gemini Vision 등)
    if file.filename.lower().endswith(".pdf"):
        print(f"📡 PDF 업로드 감지: Gemini Vision 분석 시작 - {file.filename}")
        with open(file_path, "rb") as f:
            pages_data = processing.extract_from_pdf(f.read())
            raw_text = "\n".join([p[0] for p in pages_data])
    elif file.filename.lower().endswith((".mp3", ".wav", ".m4a")):
        raw_text = processing.transcribe_audio(file_path)
    else:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    # ✅ [핵심 수정] 강의(Lecture) 확인 시 '제목'과 '유저 ID'를 함께 조회
    # 이렇게 해야 홍길동의 '소공'과 jiony의 '소공'이 분리됩니다.
    lecture = db.query(models.Lecture).filter(
        models.Lecture.title == lecture_title,
        models.Lecture.user_id == user_id
    ).first()

    # 해당 유저의 강의가 없다면 새로 생성
    if not lecture:
        lecture = models.Lecture(title=lecture_title, user_id=user_id) # ✅ user_id 반영
        db.add(lecture)
        db.commit()
        db.refresh(lecture)

    # 파일 정보 저장
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
        "user_id": user_id,
        "extracted_text": raw_text
    }

# ---------------------------------------------------------
# 2️⃣ [PUT] 텍스트 수정 (임시 저장)
# ---------------------------------------------------------
@router.put("/upload/{file_id}/text")
async def update_text(file_id: int, text_content: str = Form(...), db: Session = Depends(get_db)):
    file_rec = db.query(models.UploadedFile).filter_by(id=file_id).first()
    if not file_rec: raise HTTPException(status_code=404)
    file_rec.text_content = text_content
    db.commit()
    return {"message": "임시 저장되었습니다."}

# ---------------------------------------------------------
# 3️⃣ [POST] 파일 확정 및 소크라테스 대화 시작
# ---------------------------------------------------------
@router.post("/confirm/{file_id}")
async def confirm_file(file_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # 1. 파일 및 관련 정보 로드
    file_rec = db.query(models.UploadedFile).filter_by(id=file_id).first()
    if not file_rec:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    user = file_rec.lecture.user 
    lecture_title = file_rec.lecture.title # 강의 제목 확보

    # 2. 확정 상태 업데이트
    file_rec.is_confirmed = True
    db.commit()

    # 3. 백그라운드 임베딩 작업 시작
    background_tasks.add_task(process_file_chunks, file_id)

    # 4. 첫 질문 생성을 위한 페이지 데이터 구성
    # (추출된 페이지 정보가 있으면 활용하고, 없으면 전체 텍스트를 1페이지로 처리)
    if file_rec.page_text:
        pages_json = json.loads(file_rec.page_text)
        lecture_pages = [(item["text"], item["page"]) for item in pages_json]
    else:
        lecture_pages = [(file_rec.text_content, 1)]

    # 5. AI 첫 질문 생성 (강의명 포함)
    first_question = processing.generate_initial_question(
        lecture_pages=lecture_pages,
        dept=user.department,
        grade=user.grade,
        lecture_title=lecture_title
    )

    # 6. 새 대화 세션 생성
    new_session = models.ChatSession(
        lecture_id=file_rec.lecture_id,
        file_id=file_rec.id,
        title=f"{lecture_title} - {os.path.basename(file_rec.file_url)}"
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    # 7. 첫 AI 메시지 저장
    ai_message = models.ChatMessage(
        session_id=new_session.id,
        role="assistant",
        content=first_question
    )
    db.add(ai_message)
    db.commit()

    return {
        "message": "확정 완료! AI 튜터 Qureka가 첫 질문을 보냈습니다.",
        "session_id": new_session.id,
        "first_question": first_question
    }