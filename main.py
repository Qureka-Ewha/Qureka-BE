from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Qureka Backend Server is Running!"}