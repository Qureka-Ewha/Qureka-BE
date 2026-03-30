from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from sqlalchemy import select
from ..database import get_db
from .. import models, auth
from ..services import processing

router = APIRouter()


# ---------------------------------------------------------
# 1️⃣ AI 튜터와 대화하기
# ---------------------------------------------------------
@router.post("/talk/{session_id}")
async def talk_to_qureka(
    session_id: int,
    user_message: str = Form(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    # 1. 세션 존재 여부 확인
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(
            status_code=404,
            detail="대화 세션을 찾을 수 없습니다."
        )

    # 2. 본인 강의 세션인지 확인
    if session.lecture.user_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="접근 권한이 없습니다."
        )

    # 3. 유저 메시지 저장
    new_user_msg = models.ChatMessage(
        session_id=session.id,
        role="user",
        content=user_message
    )
    db.add(new_user_msg)
    db.commit()

    # 4. 질문 임베딩 생성
    query_vector = processing.get_embeddings([user_message])[0]

    # 5. 관련 chunk 검색
    relevant_chunks = db.scalars(
        select(models.LectureChunk)
        .filter(models.LectureChunk.lecture_id == session.lecture_id)
        .order_by(models.LectureChunk.embedding.l2_distance(query_vector))
        .limit(3)
    ).all()

    # 6. context 생성
    context_text = "\n\n".join(
        [f"[슬라이드 {c.page_number}]: {c.content}" for c in relevant_chunks]
    )

    # 7. 최근 대화 기록
    history_msgs = db.query(models.ChatMessage).filter(
        models.ChatMessage.session_id == session_id
    ).order_by(models.ChatMessage.id.desc()).limit(6).all()

    history_msgs.reverse()

    chat_history_str = "\n".join(
        [f"{m.role}: {m.content}" for m in history_msgs]
    )

    # 전체 텍스트를 chunk로 분할
    lecture_pages = processing.chunk_text_with_page([(context_text, 1)])

    # 핵심 chunk 선택
    selected_context = processing.select_key_chunks(lecture_pages)

    # 8. AI 응답 생성
    ai_reply = processing.generate_chat_response(
        context_text=selected_context,
        chat_history=chat_history_str,
        dept=current_user.department,
        grade=current_user.grade
    )

    # 9. AI 메시지 저장
    new_ai_msg = models.ChatMessage(
        session_id=session.id,
        role="assistant",
        content=ai_reply
    )

    db.add(new_ai_msg)
    db.commit()

    return {
        "ai_reply": ai_reply,
        "source_pages": list(
            set([c.page_number for c in relevant_chunks if c.page_number])
        )
    }


# ---------------------------------------------------------
# 2️⃣ 대화 세션 목록 조회
# ---------------------------------------------------------
@router.get("/sessions")
def get_chat_sessions(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    sessions = db.query(models.ChatSession).join(models.Lecture).filter(
        models.Lecture.user_id == current_user.id
    ).order_by(models.ChatSession.created_at.desc()).all()

    return [
        {
            "session_id": s.id,
            "title": s.title,
            "lecture_id": s.lecture_id,
            "created_at": s.created_at
        }
        for s in sessions
    ]


# ---------------------------------------------------------
# 3️⃣ 특정 세션 대화 내역 조회
# ---------------------------------------------------------
@router.get("/history/{session_id}")
def get_chat_history(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    # 세션 확인
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(
            status_code=404,
            detail="세션을 찾을 수 없습니다."
        )

    # 본인 세션인지 확인
    if session.lecture.user_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="접근 권한이 없습니다."
        )

    # 메시지 조회
    messages = db.query(models.ChatMessage).filter(
        models.ChatMessage.session_id == session_id
    ).order_by(models.ChatMessage.created_at.asc()).all()

    return [
        {
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at
        }
        for m in messages
    ]

# ---------------------------------------------------------
# 4️⃣ 세션 제목 수정
# ---------------------------------------------------------
@router.patch("/sessions/{session_id}")
def update_session_title(
    session_id: int,
    title: str = Form(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="세션 없음")

    if session.lecture.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="권한 없음")

    session.title = title
    db.commit()

    return {"message": "제목 수정 완료"}

# ---------------------------------------------------------
# 5️⃣ 세션 삭제
# ---------------------------------------------------------
@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="세션 없음")

    if session.lecture.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="권한 없음")

    db.query(models.ChatMessage).filter(
        models.ChatMessage.session_id == session_id
    ).delete()

    db.delete(session)
    db.commit()

    return {"message": "삭제 완료"}