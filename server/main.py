from fastapi import FastAPI
from server.api.routes import router as match_router

app = FastAPI(title="IR-VIS Face Recognition API")
app.include_router(match_router)