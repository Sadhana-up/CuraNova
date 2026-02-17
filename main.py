from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn


app = FastAPI(
    title="CuraNova",
    description="Medical AI Backend ",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthRequest(BaseModel):
    query: str
    patient_id: Optional[str] = None


class HealthResponse(BaseModel):
    response: str
    status: str


# Routes
@app.get("/")
async def root():
    return {"message": "CuraNova API is running successfully!"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=HealthResponse)
async def analyze(data: HealthRequest):
    try:
        # TODO: Connect your Med model here
        model_response = f"Processed query: {data.query}"

        return HealthResponse(
            response=model_response,
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run Server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
