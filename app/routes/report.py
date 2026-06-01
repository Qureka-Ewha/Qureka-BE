from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload

from ..database import get_db
from .. import models, auth
from ..services import report

router = APIRouter()


@router.get("/report/{session_id}")
def get_learning_report(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):

    # 1. 세션 조회
    session = (
        db.query(models.ChatSession)
        .options(joinedload(models.ChatSession.lecture))
        .filter(models.ChatSession.id == session_id)
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="세션을 찾을 수 없습니다."
        )

    # 2. 권한 확인 (현재 로그인 사용자의 강의인지)
    if session.lecture.user_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="접근 권한이 없습니다."
        )

    # 3. 리포트 생성을 위한 전체 대화 조회
    messages = db.query(models.ChatMessage).filter(
        models.ChatMessage.session_id == session_id
    ).order_by(models.ChatMessage.id).all()

    # 4. role과 응답 지연 계산용 시간 정보 포함
    chat_history = [
        {
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in messages
    ]

    # 5. 리포트 생성
    result = report.generate_learning_report(chat_history)

    return result