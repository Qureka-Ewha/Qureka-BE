import fitz  # PyMuPDF의 이름입니다.

def extract_text_from_pdf(file_content: bytes):
    """
    PDF 파일의 바이너리 데이터를 받아서 텍스트만 추출하는 함수
    """
    text_result = ""
    
    # 1. PDF 파일을 엽니다 (메모리 상에서)
    doc = fitz.open(stream=file_content, filetype="pdf")
    
    # 2. 페이지를 한 장씩 넘기며 글자를 뽑습니다.
    for page_num, page in enumerate(doc):
        text = page.get_text()
        text_result += f"--- Page {page_num + 1} ---\n{text}\n"
        
    return text_result