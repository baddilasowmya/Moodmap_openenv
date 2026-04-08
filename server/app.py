from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"message": "MoodMap Env Running 🚀"}


# 👇 ADD THIS
class Input(BaseModel):
    text: str

@app.post("/predict")
def predict(data: Input):
    return {"result": f"You said: {data.text}"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()