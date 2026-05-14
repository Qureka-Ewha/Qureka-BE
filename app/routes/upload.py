from fastapi import APIRouter, UploadFile, File, Depends, BackgroundTasks, HTTPException, Form
from sqlalchemy.orm import Session, joinedload
from ..database import get_db, SessionLocal
from .. import models, auth
from ..services import processing
from typing import List, Tuple

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
                raw = processing.transcribe_audio(path)
                file_rec.transcript_raw = raw
                if processing.is_stt_service_error(raw):
                    file_rec.text_content = raw
                else:
                    file_rec.text_content = None
            elif key.endswith(".txt"):
                file_rec.page_text = None
                file_rec.text_content = processing.read_uploaded_txt_file(path)
            else:
                file_rec.text_content = "[지원하지 않는 파일 형식입니다]"
                file_rec.page_text = None
        except Exception as e:
            file_rec.page_text = None
            file_rec.transcript_raw = None
            file_rec.text_content = f"[텍스트 추출 오류] {e}"

        db.commit()

        if file_rec.upload_group_id and file_rec.lecture_id:
            try_refine_audios_in_group(
                file_rec.upload_group_id,
                file_rec.lecture_id,
            )
    finally:
        db.close()


def _peer_reference_corpus(peers: List[models.UploadedFile]) -> str:
    """묶음 내 음성이 아닌 파일 중, 추출에 성공한 본문만 합침 (STT 교정용)."""
    parts: List[str] = []
    for p in peers:
        if processing.is_audio_file_url(p.file_url):
            continue
        tc = (p.text_content or "").strip()
        if not tc or _looks_like_extraction_error(tc):
            continue
        name = p.original_name or "자료"
        parts.append(f"=== {name} ===\n{tc}\n")
    return "\n".join(parts).strip()


def try_refine_audios_in_group(upload_group_id: str | None, lecture_id: int) -> None:
    """강의 자료 텍스트가 준비된 뒤, 묶음 내 음성 STT를 자료 근거로만 교정해 text_content에 반영."""
    if not upload_group_id:
        return
    db = SessionLocal()
    try:
        peers = (
            db.query(models.UploadedFile)
            .filter(
                models.UploadedFile.lecture_id == lecture_id,
                models.UploadedFile.upload_group_id == upload_group_id,
            )
            .order_by(models.UploadedFile.id.asc())
            .all()
        )
        if not peers:
            return
        ref = _peer_reference_corpus(peers)
        for p in peers:
            if not processing.is_audio_file_url(p.file_url):
                continue
            db.refresh(p)
            if p.text_content is not None:
                continue
            raw = (p.transcript_raw or "").strip()
            if not raw or processing.is_stt_service_error(p.transcript_raw):
                continue
            if not ref:
                p.text_content = raw
                db.commit()
                continue
            refined = processing.refine_stt_with_lecture_docs(raw, ref)
            p.text_content = refined if refined else raw
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
                fallback = [(file_rec.text_content or "", None)]
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


def _group_peers(db: Session, file_rec: models.UploadedFile) -> List[models.UploadedFile]:
    """동일 업로드 묶음(또는 레거시 단일 파일)에 속한 UploadedFile 목록."""
    q = db.query(models.UploadedFile).filter(
        models.UploadedFile.lecture_id == file_rec.lecture_id,
    )
    if file_rec.upload_group_id:
        q = q.filter(models.UploadedFile.upload_group_id == file_rec.upload_group_id)
    else:
        q = q.filter(models.UploadedFile.id == file_rec.id)
    return q.order_by(models.UploadedFile.id.asc()).all()


def _batch_source_kind(peers: List[models.UploadedFile]) -> processing.SourceKind:
    for p in peers:
        if p.file_url.lower().endswith(".pdf"):
            return "pdf"
    return "transcript"


def build_combined_lecture_pages(
    peers: List[models.UploadedFile],
) -> List[Tuple[str, int | None]]:
    """묶음 내 모든 파일 텍스트를 페이지 리스트로 병합 (PDF는 페이지 단위, 그 외는 블록)."""
    pages: List[Tuple[str, int | None]] = []
    page_counter = 1
    for f in peers:
        key = f.file_url.lower()
        name = f.original_name or "파일"
        if key.endswith(".pdf") and f.page_text:
            try:
                arr = json.loads(f.page_text)
                added_pages = 0
                for item in arr:
                    t = (item.get("text") or "").strip()
                    if not t:
                        continue
                    pages.append((t, page_counter))
                    page_counter += 1
                    added_pages += 1
                if added_pages > 0:
                    continue
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                pass
        body = (f.text_content or "").strip()
        if body:
            pages.append((f"[자료: {name}]\n{body}", None))
    if not pages:
        return [("", None)]
    return pages


def _find_existing_batch_session(
    db: Session,
    lecture_id: int,
    upload_group_id: str | None,
    single_peer_id: int | None,
) -> models.ChatSession | None:
    if upload_group_id:
        return (
            db.query(models.ChatSession)
            .filter(
                models.ChatSession.lecture_id == lecture_id,
                models.ChatSession.upload_group_id == upload_group_id,
            )
            .first()
        )
    if single_peer_id is not None:
        return (
            db.query(models.ChatSession)
            .filter(
                models.ChatSession.lecture_id == lecture_id,
                models.ChatSession.file_id == single_peer_id,
                models.ChatSession.upload_group_id.is_(None),
            )
            .first()
        )
    return None


@router.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    lecture_title: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):

    uid = current_user.id

    def _orig_lower(uf: UploadFile) -> str:
        return (os.path.basename(uf.filename or "") or "").lower()

    has_audio = any(
        _orig_lower(f).endswith((".mp3", ".wav", ".m4a")) for f in files
    )
    has_pdf = any(_orig_lower(f).endswith(".pdf") for f in files)
    if has_audio and not has_pdf:
        raise HTTPException(
            status_code=400,
            detail=(
                "음성 파일은 동일 요청에서 PDF 강의 자료를 함께 업로드해야 합니다. "
                "STT 결과는 업로드한 강의 자료에만 근거하여 자동 교정된 뒤에 표시됩니다."
            ),
        )

    # 같은 강의명 있으면 재사용
    lecture = db.query(models.Lecture).filter(
        models.Lecture.title == lecture_title,
        models.Lecture.user_id == uid,
    ).first()

    # 없으면 새 생성
    if not lecture:
        lecture = models.Lecture(
            title=lecture_title,
            user_id=uid
        )
        db.add(lecture)
        db.commit()
        db.refresh(lecture)

    uploaded_results = []

    upload_group_id = str(uuid.uuid4())

    # 여러 파일 반복 처리
    for file in files:

        orig_name = os.path.basename(file.filename or "") or "upload"

        _, ext = os.path.splitext(orig_name)

        stored_name = f"{uuid.uuid4().hex}{ext}"

        file_path = os.path.join(UPLOAD_DIR, stored_name)

        # 파일 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        tl = orig_name.lower()

        # 지원 파일 형식 검사
        if not (
            tl.endswith(".pdf")
            or tl.endswith((".mp3", ".wav", ".m4a", ".txt"))
        ):
            try:
                os.remove(file_path)
            except OSError:
                pass

            continue

        # DB 저장
        new_file = models.UploadedFile(
            lecture_id=lecture.id,
            file_url=file_path,
            file_type=file.content_type,
            original_name=orig_name,
            text_content=None,
            is_confirmed=False,
            upload_group_id=upload_group_id,
        )

        db.add(new_file)
        db.commit()
        db.refresh(new_file)

        # txt는 OCR/STT 없이 즉시 본문 반영 → 수정 화면 바로 표시
        if tl.endswith(".txt"):
            try:
                new_file.text_content = processing.read_uploaded_txt_file(file_path)
                new_file.page_text = None
                db.commit()
                db.refresh(new_file)
                extraction_pending = False
            except Exception as e:
                new_file.page_text = None
                new_file.text_content = f"[텍스트 추출 오류] {e}"
                db.commit()
                db.refresh(new_file)
                extraction_pending = False
            print(f"txt 즉시 로드(file_id={new_file.id}): {orig_name}")
        else:
            background_tasks.add_task(
                extract_media_to_file_record,
                new_file.id,
            )
            extraction_pending = True
            print(
                f"업로드 수신 후 백그라운드 추출 예약(file_id={new_file.id}): {orig_name}"
            )

        result_entry = {
            "file_id": new_file.id,
            "filename": orig_name,
            "upload_group_id": upload_group_id,
            "extraction_pending": extraction_pending,
        }
        if tl.endswith(".txt"):
            result_entry["extracted_text"] = new_file.text_content
        uploaded_results.append(result_entry)

    try_refine_audios_in_group(upload_group_id, lecture.id)

    return {
        "lecture_id": lecture.id,
        "user_id": uid,
        "upload_group_id": upload_group_id,
        "uploaded_files": uploaded_results,
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
    is_audio = processing.is_audio_file_url(file_rec.file_url)
    refinement_pending = bool(
        is_audio
        and pending
        and (file_rec.transcript_raw or "").strip()
        and not processing.is_stt_service_error(file_rec.transcript_raw)
    )
    return {
        "file_id": file_id,
        "upload_group_id": file_rec.upload_group_id,
        "is_confirmed": file_rec.is_confirmed,
        "pending": pending,
        "failure": err,
        "refinement_pending": refinement_pending,
        "extracted_text": None if pending else file_rec.text_content,
    }

# ---------------------------------------------------------
# 동일 POST 업로드 묶음 상태 (파일별 검토·확정 진행률)
# ---------------------------------------------------------
@router.get("/upload/batch/{upload_group_id}")
def get_upload_batch_status(
    upload_group_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    files = (
        db.query(models.UploadedFile)
        .join(models.Lecture)
        .filter(
            models.Lecture.user_id == current_user.id,
            models.UploadedFile.upload_group_id == upload_group_id,
        )
        .order_by(models.UploadedFile.id.asc())
        .all()
    )
    if not files:
        raise HTTPException(status_code=404, detail="해당 업로드 묶음을 찾을 수 없습니다.")
    return {
        "upload_group_id": upload_group_id,
        "lecture_id": files[0].lecture_id,
        "total": len(files),
        "all_confirmed": all(f.is_confirmed for f in files),
        "files": [
            {
                "file_id": f.id,
                "filename": f.original_name,
                "extraction_pending": f.text_content is None,
                "extraction_failure": _looks_like_extraction_error(f.text_content),
                "refinement_pending": bool(
                    processing.is_audio_file_url(f.file_url)
                    and f.text_content is None
                    and (f.transcript_raw or "").strip()
                    and not processing.is_stt_service_error(f.transcript_raw)
                ),
                "is_confirmed": f.is_confirmed,
            }
            for f in files
        ],
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
# 3️⃣ [POST] 파일 확정 — 동일 업로드 묶음은 모두 확정된 뒤에만 튜터·첫 질문 시작
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

    peers = _group_peers(db, file_rec)
    for p in peers:
        if p.text_content is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"파일 '{p.original_name or p.id}' 분석이 아직 끝나지 않았습니다. "
                    f"(GET /files/upload/{p.id}/extraction)"
                ),
            )
        if _looks_like_extraction_error(p.text_content):
            raise HTTPException(
                status_code=409,
                detail=f"추출에 실패한 파일이 있어 확정할 수 없습니다: {p.original_name}",
            )

    user = file_rec.lecture.user
    lecture_title = file_rec.lecture.title
    gid = file_rec.upload_group_id
    single_peer_id = peers[0].id if len(peers) == 1 and not gid else None

    existing = _find_existing_batch_session(
        db, file_rec.lecture_id, gid, single_peer_id
    )
    if existing:
        first_ai = (
            db.query(models.ChatMessage)
            .filter(models.ChatMessage.session_id == existing.id)
            .order_by(models.ChatMessage.id.asc())
            .first()
        )
        fq = (
            first_ai.content
            if first_ai and first_ai.role == "assistant"
            else ""
        )
        sk: processing.SourceKind = (
            existing.source_kind
            if existing.source_kind in ("pdf", "transcript")
            else _batch_source_kind(peers)
        )
        return {
            "status": "batch_complete",
            "message": "이미 모든 파일이 확정되어 튜터를 시작했습니다.",
            "session_id": existing.id,
            "first_question": fq,
            "source_kind": sk,
            "upload_group_id": gid,
        }

    if not file_rec.is_confirmed:
        file_rec.is_confirmed = True
        db.commit()

    peers = _group_peers(db, file_rec)
    pending = [p for p in peers if not p.is_confirmed]
    if pending:
        return {
            "status": "awaiting_other_files",
            "message": (
                "같은 업로드 묶음에 아직 검토·확정하지 않은 파일이 있습니다. "
                "나머지 파일도 수정 후 확정해 주세요."
            ),
            "upload_group_id": gid,
            "total_in_group": len(peers),
            "confirmed_count": len([p for p in peers if p.is_confirmed]),
            "pending_files": [
                {"file_id": p.id, "filename": p.original_name} for p in pending
            ],
        }

    for p in peers:
        background_tasks.add_task(process_file_chunks, p.id)

    lecture_pages = build_combined_lecture_pages(peers)
    source_kind = _batch_source_kind(peers)
    first_question = processing.generate_initial_question(
        lecture_pages=lecture_pages,
        dept=user.department,
        grade=user.grade,
        lecture_title=lecture_title,
        source_kind=source_kind,
    )

    session_file_id = None
    session_upload_gid = gid
    if len(peers) == 1 and not gid:
        session_file_id = peers[0].id
        session_upload_gid = None

    new_session = models.ChatSession(
        lecture_id=file_rec.lecture_id,
        file_id=session_file_id,
        title=f"{lecture_title} · 새 대화",
        source_kind=source_kind,
        upload_group_id=session_upload_gid,
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    ai_message = models.ChatMessage(
        session_id=new_session.id,
        role="assistant",
        content=first_question,
    )
    db.add(ai_message)
    db.commit()

    return {
        "status": "batch_complete",
        "message": "확정 완료! 업로드한 모든 자료를 반영해 첫 질문을 보냈습니다.",
        "session_id": new_session.id,
        "first_question": first_question,
        "source_kind": source_kind,
        "upload_group_id": gid,
    }