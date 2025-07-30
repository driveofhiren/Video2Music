from fastapi import FastAPI, UploadFile, File, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import uuid
import asyncio
from pathlib import Path
# from final import process_upload, start_live_processing
from final import start_live_processing
import cv2
import time
from typing import Dict
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocketDisconnect

app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# Create uploads directory if not exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# WebSocket connections management
active_connections: Dict[str, WebSocket] = {}

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
    asyncio.create_task(process_upload(file_path, is_video, broadcast_audio))
    return {"message": "Processing started - audio will stream to connected clients"}

@app.post("/start-live")
async def handle_start_live():
    asyncio.create_task(start_live_processing(broadcast_audio))
    return {"message": "Live processing started"}

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    active_connections[connection_id] = websocket
    try:
        while True:
            # Try to receive a message to keep connection alive
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle ping/pong if needed
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_text("ping")
                except:
                    break  # Connection is dead
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket receive error: {e}")
                break
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        active_connections.pop(connection_id, None)
        print(f"Client {connection_id} disconnected")

async def broadcast_audio(audio_data: bytes):
    if not active_connections:
        return
        
    # Ensure the audio data is in the correct format
    if not isinstance(audio_data, bytes):
        audio_data = bytes(audio_data)
    
    # Add WAV header if needed (example for 16-bit 44.1kHz stereo)
    if len(audio_data) > 0:
        # Simple WAV header for raw PCM data
        header = bytearray()
        header.extend(b'RIFF')
        header.extend((len(audio_data) + 36).to_bytes(4, byteorder='little'))  # File size
        header.extend(b'WAVEfmt ')
        header.extend((16).to_bytes(4, byteorder='little'))  # Subchunk size
        header.extend((1).to_bytes(2, byteorder='little'))   # Audio format (PCM)
        header.extend((2).to_bytes(2, byteorder='little'))   # Channels
        header.extend((44100).to_bytes(4, byteorder='little'))  # Sample rate
        header.extend((176400).to_bytes(4, byteorder='little'))  # Byte rate
        header.extend((4).to_bytes(2, byteorder='little'))    # Block align
        header.extend((16).to_bytes(2, byteorder='little'))   # Bits per sample
        header.extend(b'data')
        header.extend(len(audio_data).to_bytes(4, byteorder='little'))
        
        full_audio = bytes(header) + audio_data
    else:
        full_audio = audio_data
    
    # Send to all active connections
    for connection_id, websocket in list(active_connections.items()):
        try:
            await websocket.send_bytes(full_audio)
        except Exception as e:
            print(f"Error broadcasting to {connection_id}: {e}")
            try:
                await websocket.close()
            except:
                pass
            active_connections.pop(connection_id, None)