from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from .. import models, auth, database

router = APIRouter()

# ---------------------------------------------------------
# 1️⃣ 내 정보 불러오기 API
# ---------------------------------------------------------
@router.get("/me")
def get_my_info(current_user: models.User = Depends(auth.get_current_user)):
    """
    현재 로그인한 사용자의 정보를 반환합니다.
    (프론트엔드에서 수정 페이지를 열 때, 원래 내 이름을 빈칸에 채워주기 위해 사용)
    """
    return {
        "email": current_user.email,
        "nickname": current_user.nickname,
        "department": current_user.department,
        "grade": current_user.grade
    }

# ---------------------------------------------------------
# 2️⃣ 내 정보 수정하기 API
# ---------------------------------------------------------
@router.patch("/me")
def update_my_info(
    new_nickname: str = Form(None), 
    new_password: str = Form(None), 
    db: Session = Depends(database.get_db), 
    current_user: models.User = Depends(auth.get_current_user)
):
    """
    사용자가 입력한 새로운 닉네임이나 비밀번호로 DB를 수정합니다.
    """
    # 1. 닉네임 변경이 요청된 경우
    if new_nickname:
        current_user.nickname = new_nickname
        
    # 2. 비밀번호 변경이 요청된 경우 (반드시 암호화해서 저장!)
    if new_password:
        current_user.hashed_password = auth.get_password_hash(new_password)
    
    # 3. DB에 변경사항 저장
    db.commit()
    
    return {"message": "개인정보가 성공적으로 수정되었습니다."}