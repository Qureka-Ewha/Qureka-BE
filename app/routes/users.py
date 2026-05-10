from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from .. import models, auth, database

router = APIRouter()

# ---------------------------------------------------------
# 1️⃣ 내 정보 불러오기 API (기존 유지)
# ---------------------------------------------------------
@router.get("/me")
def get_my_info(current_user: models.User = Depends(auth.get_current_user)):
    return {
        "email": current_user.email,
        "nickname": current_user.nickname,
        "department": current_user.department,
        "grade": current_user.grade
    }

# ---------------------------------------------------------
# 2️⃣ 내 정보 수정하기 API (보안 강화 & 필드 추가!)
# ---------------------------------------------------------
@router.patch("/me")
def update_my_info(
    current_password: str = Form(...),
    new_nickname: Optional[str] = Form(default=None),
    new_department: Optional[str] = Form(default=None),
    new_grade: Optional[str] = Form(default=None),
    new_password: Optional[str] = Form(default=None),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    # 🚨 가장 중요: 현재 비밀번호가 맞는지부터 확인!
    if not auth.verify_password(current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="현재 비밀번호가 일치하지 않습니다.")

    # 비밀번호가 맞다면, 요청 들어온 정보들만 쏙쏙 업데이트
    if new_nickname:
        current_user.nickname = new_nickname
    if new_department:
        current_user.department = new_department
    if new_grade is not None and str(new_grade).strip():
        try:
            current_user.grade = int(new_grade)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="학년은 정수 형식으로 보내주세요.",
            )
    if new_password:
        current_user.hashed_password = auth.get_password_hash(new_password)
    
    db.commit()
    
    return {"message": "개인정보가 성공적으로 수정되었습니다."}