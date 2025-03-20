from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from router.eye_tracker import eye_tracker

load_dotenv()

host = os.getenv("SERVER_HOST")
port = int(os.getenv("SERVER_PORT"))

app = FastAPI()

# CORS 미들웨어 추가 (외부 요청 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def root() :
    return print('서버 연결 됨')

@app.get('/live_stream')
def live_stream() :
    return StreamingResponse(eye_tracker(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__" :
    uvicorn.run("main:app", host=f"{host}", port=port, reload=True)