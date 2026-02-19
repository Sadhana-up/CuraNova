import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the 'static' directory to serve HTML/CSS/JS
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("✨ CuraNova Client Running")
    print("👉 Open: http://localhost:8080")
    uvicorn.run(app, host="127.0.0.1", port=8080)