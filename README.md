# Leonar-Draw Project

Leonar-Draw는 별도의 장비 없이 웹캠과 눈동자 움직임만으로 그림을 그릴 수 있는 시스템입니다.

이 저장소는 그 중 백엔드 서버를 담당하는 FastAPI 기반 프로젝트입니다.


## 🎯 주요 기능

👁️ 눈동자 추적 기능
MediaPipe를 활용하여 실시간으로 눈동자 위치를 인식합니다.

🖱️ 마우스 입력 연동
눈의 움직임을 화면상의 좌표로 변환하여 마우스 커서처럼 사용할 수 있습니다.

🧠 AI 기반 좌표 보정
투시 변환, 동적 가중치, 다항 회귀 등을 통해 부드럽고 정확한 좌표 예측이 가능합니다.

🚀 FastAPI 백엔드
빠르고 직관적인 API 서버로 실시간 통신을 처리합니다.


## 📁 프로젝트 구조
```bash

FastAPI/
├── router/            # API 라우터 정의
│   └── __init__.py
├── main.py            # FastAPI 앱 실행 파일
├── requirements.txt   # 필요 라이브러리 목록
└── .gitignore
```

## ⚙️ 실행 방법
1. 저장소 클론
```bash
git clone https://github.com/Leonar-Draw/FastAPI.git
cd FastAPI
```

2. 가상환경 설정 (선택)
```bash
python -m venv venv
source venv/bin/activate  # 윈도우: venv\Scripts\activate
```

3. 라이브러리 설치
```bash
pip install -r requirements.txt
```

4. 서버 실행
```bash
uvicorn main:app --reload
```

5. API 문서 확인
- Swagger 문서: http://localhost:8000/docs
- ReDoc 문서: http://localhost:8000/redoc

## 🖼️ 데모 페이지
http://localhost:8000/draw
→ 눈으로 그림을 그리는 페이지 제공 (프론트와 연결 시 적용)

## 👥 기여 방법
Pull Request는 언제나 환영입니다!

코드 컨벤션 및 기능 분리에 유의하여 기여해주세요.

## 📄 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.
