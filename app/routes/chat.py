from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from sqlalchemy import select
from ..database import get_db
from .. import models, auth
from ..services import processing

router = APIRouter()

@router.post("/talk/{session_id}")
async def talk_to_qureka(
    session_id: int,
    user_message: str = Form(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    # 1. 세션 존재 여부 및 권한 확인
    session = db.query(models.ChatSession).filter(models.ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="대화 세션을 찾을 수 없습니다.")
    
    # 2. 유저 메시지 저장
    new_user_msg = models.ChatMessage(session_id=session.id, role="user", content=user_message)
    db.add(new_user_msg)
    db.commit()

    # 3. RAG: 질문과 가장 관련 있는 강의 조각 찾기
    # 사용자 답변을 벡터로 변환
    query_vector = processing.get_embeddings([user_message])[0]
    
    # DB에서 가장 유사한 강의 조각 3개를 찾습니다.
    relevant_chunks = db.scalars(
        select(models.LectureChunk)
        .filter(models.LectureChunk.lecture_id == session.lecture_id)
        .order_by(models.LectureChunk.embedding.l2_distance(query_vector)) # 벡터 거리순 정렬
        .limit(3)
    ).all()
    
    # 찾은 내용을 하나의 텍스트로 합칩니다.
    context_text = "\n\n".join([f"[슬라이드 {c.page_number}]: {c.content}" for c in relevant_chunks])

    # 4. 이전 대화 기록(History) 가져오기 (최근 5개 정도)
    history_msgs = db.query(models.ChatMessage).filter(
        models.ChatMessage.session_id == session_id
    ).order_by(models.ChatMessage.id.desc()).limit(6).all()
    
    history_msgs.reverse()
    chat_history_str = "\n".join([f"{m.role}: {m.content}" for m in history_msgs])

    # 5. Qureka 응답 생성 (학과/학년 정보 주입)
    ai_reply = processing.generate_chat_response(
        context_text=context_text,
        chat_history=chat_history_str,
        dept=current_user.department,
        grade=current_user.grade
    )

    # 6. AI 메시지 저장
    new_ai_msg = models.ChatMessage(session_id=session.id, role="assistant", content=ai_reply)
    db.add(new_ai_msg)
    db.commit()

    return {
        "ai_reply": ai_reply,
        "source_pages": list(set([c.page_number for c in relevant_chunks if c.page_number]))
    }