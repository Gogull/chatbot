from fastapi import FastAPI

app = FastAPI(title="Dummy FastAPI on Vercel")

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI on Vercel!"}

