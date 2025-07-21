from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import uuid
import asyncio
from pathlib import Path
from final import process_upload, start_live_processing

app = FastAPI()

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# Create uploads directory if not exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def handle_upload(file: UploadFile = File(...)):
    # Save uploaded file
    file_ext = Path(file.filename).suffix
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Determine if it's video or image
    is_video = file.content_type.startswith("video/")
    
    # Process in background
    asyncio.create_task(process_upload(file_path, is_video))
    return {"message": "Processing started - check your audio output"}

@app.post("/start-live")
async def start_live():
    # Start live processing in background
    asyncio.create_task(start_live_processing())
    return {"message": "Live processing started - check your audio output"}