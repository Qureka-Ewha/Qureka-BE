from fastapi import APIRouter, UploadFile, File, Depends, BackgroundTasks, HTTPException, Form
from sqlalchemy.orm import Session, joinedload
from ..database import get_db, SessionLocal
from .. import models, auth
from ..services import processing
import shutil
import os
import json
import uuid

router = APIRouter()

# 업로드 경로 설정
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------
# 업로드 직후: 디스크에 저장만 한 뒤 텍스트/페이지를 백그라운드에서 채움
# ---------------------------------------------------------
def extract_media_to_file_record(file_id: int):
    db = SessionLocal()
    try:
        file_rec = db.query(models.UploadedFile).filter_by(id=file_id).first()
        if not file_rec:
            return

        path = file_rec.file_url
        key = path.lower()

        try:
            if key.endswith(".pdf"):
                with open(path, "rb") as f:
                    pages_data = processing.extract_from_pdf(f.read())
                raw_text = "\n".join(t for t, _ in pages_data)
                file_rec.text_content = raw_text
                file_rec.page_text = json.dumps(
                    [{"text": t, "page": pn} for t, pn in pages_data],
                    ensure_ascii=False,
                )
            elif key.endswith((".mp3", ".wav", ".m4a")):
                file_rec.page_text = None
                file_rec.text_content = processing.transcribe_audio(path)
            else:
                file_rec.text_content = "[지원하지 않는 파일 형식입니다]"
                file_rec.page_text = None
        except Exception as e:
            file_rec.page_text = None
            file_rec.text_content = f"[텍스트 추출 오류] {e}"

        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------
# 확정 시: 저장된 페이지 데이터로 청킹 후 임베딩 (PDF 재 OCR 방지)
# ---------------------------------------------------------
def process_file_chunks(file_id: int):
    db = SessionLocal()
    try:
        file_rec = db.query(models.UploadedFile).filter_by(id=file_id).first()
        if not file_rec or not file_rec.text_content:
            return

        db.query(models.LectureChunk).filter(
            models.LectureChunk.file_id == file_id
        ).delete(synchronize_session=False)

        key = file_rec.file_url.lower()
        chunks_with_page = None

        if key.endswith(".pdf") and file_rec.page_text:
            try:
                pages_json = json.loads(file_rec.page_text)
                pages_data = [(item["text"], item["page"]) for item in pages_json]
                chunks_with_page = processing.chunk_text_with_page(pages_data)
            except (json.JSONDecodeError, KeyError, TypeError):
                chunks_with_page = None

        if chunks_with_page is None:
            if key.endswith(".pdf"):
                fallback = (
                    [(file_rec.text_content, 1)]
                    if file_rec.text_content
                    else []
                )
            else:
                fallback = [(file_rec.text_content or "", 1)]
            if not fallback or not (fallback[0][0] or "").strip():
                return
            chunks_with_page = processing.chunk_text_with_page(fallback)

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

def _looks_like_extraction_error(text: str | None) -> bool:
    return bool(text) and (
        text.startswith("[텍스트 추출 오류]")
        or text.startswith("[지원하지 않는 파일 형식입니다]")
    )


@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    lecture_title: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    orig_name = os.path.basename(file.filename or "") or "upload"
    _, ext = os.path.splitext(orig_name)
    stored_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, stored_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    tl = orig_name.lower()
    if not (
        tl.endswith(".pdf")
        or tl.endswith((".mp3", ".wav", ".m4a"))
    ):
        try:
            os.remove(file_path)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    uid = current_user.id
    lecture = db.query(models.Lecture).filter(
        models.Lecture.title == lecture_title,
        models.Lecture.user_id == uid,
    ).first()

    if not lecture:
        lecture = models.Lecture(title=lecture_title, user_id=uid)
        db.add(lecture)
        db.commit()
        db.refresh(lecture)

    new_file = models.UploadedFile(
        lecture_id=lecture.id,
        file_url=file_path,
        file_type=file.content_type,
        text_content=None,
        is_confirmed=False,
    )
    db.add(new_file)
    db.commit()
    db.refresh(new_file)

    background_tasks.add_task(extract_media_to_file_record, new_file.id)
    print(f"📡 업로드 수신 후 백그라운드 추출 예약(file_id={new_file.id}): {orig_name}")

    return {
        "file_id": new_file.id,
        "lecture_id": lecture.id,
        "user_id": uid,
        "extracted_text": None,
        "extraction_pending": True,
    }


@router.get("/upload/{file_id}/extraction")
def get_extraction_status(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    file_rec = (
        db.query(models.UploadedFile)
        .options(joinedload(models.UploadedFile.lecture))
        .filter(models.UploadedFile.id == file_id)
        .first()
    )
    if not file_rec:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    if file_rec.lecture.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")

    pending = file_rec.text_content is None
    err = (
        False
        if pending
        else _looks_like_extraction_error(file_rec.text_content)
    )
    return {
        "file_id": file_id,
        "pending": pending,
        "failure": err,
        "extracted_text": None if pending else file_rec.text_content,
    }

# ---------------------------------------------------------
# 2️⃣ [PUT] 텍스트 수정 (임시 저장)
# ---------------------------------------------------------
@router.put("/upload/{file_id}/text")
async def update_text(
    file_id: int,
    text_content: str = Form(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    file_rec = (
        db.query(models.UploadedFile)
        .options(joinedload(models.UploadedFile.lecture))
        .filter(models.UploadedFile.id == file_id)
        .first()
    )
    if not file_rec:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    if file_rec.lecture.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")

    file_rec.text_content = text_content
    db.commit()
    return {"message": "임시 저장되었습니다."}

# ---------------------------------------------------------
# 3️⃣ [POST] 파일 확정 및 소크라테스 대화 시작
# ---------------------------------------------------------
@router.post("/confirm/{file_id}")
async def confirm_file(
    file_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    file_rec = (
        db.query(models.UploadedFile)
        .options(joinedload(models.UploadedFile.lecture))
        .filter(models.UploadedFile.id == file_id)
        .first()
    )
    if not file_rec:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    if file_rec.lecture.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")

    db.refresh(file_rec)
    if file_rec.text_content is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "파일 분석이 아직 끝나지 않았습니다. "
                "잠시 후 다시 확정해 주세요. (진행 상태: GET /files/upload/{}/extraction)".format(
                    file_id,
                ),
            ),
        )

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
        title=f"{lecture_title} · 새 대화",
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