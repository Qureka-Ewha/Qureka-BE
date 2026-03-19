from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

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

    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")


    messages = db.query(models.ChatMessage).filter(
        models.ChatMessage.session_id == session_id,
        models.ChatMessage.role == "user"
    ).order_by(models.ChatMessage.id).all()


    user_messages = [m.content for m in messages]


    result = report.generate_learning_report(user_messages)

    return result