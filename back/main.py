#main.py
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
from final import start_client_video_processing, update_frame_from_client
import cv2
import time
from typing import Dict
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocketDisconnect
import uvicorn
import json

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

# WebSocket connections management
active_connections: Dict[str, WebSocket] = {}
video_connections: Dict[str, WebSocket] = {}
music_processing_task = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start-live")
async def handle_start_live():
    global music_processing_task
    
    # Cancel existing task if running
    if music_processing_task and not music_processing_task.done():
        music_processing_task.cancel()
        try:
            await music_processing_task
        except asyncio.CancelledError:
            pass
    
    # Start new music processing task
    music_processing_task = asyncio.create_task(start_client_video_processing(broadcast_audio))
    return {"message": "Live processing started - waiting for video frames"}

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


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    video_connections[connection_id] = websocket
    print(f"Video client {connection_id} connected")
    
    try:
        while True:
            try:
                # Receive frame data from client
                data = await websocket.receive_text()
                frame_data = json.loads(data)
                
                # Process the base64 encoded frame
                if 'frame' in frame_data:
                    # Extract base64 data (remove data:image/jpeg;base64, prefix)
                    base64_data = frame_data['frame'].split(',')[1]
                    
                    # Decode base64 to bytes
                    image_bytes = base64.b64decode(base64_data)
                    
                    # Update the current frame for processing
                    await update_frame_from_client(image_bytes)
                    
                    # Optional: Send acknowledgment back to client
                    await websocket.send_text(json.dumps({
                        "status": "frame_received",
                        "timestamp": frame_data.get('timestamp', time.time())
                    }))
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
            except Exception as e:
                print(f"Video processing error: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))
                
    except Exception as e:
        print(f"Video WebSocket error: {e}")
    finally:
        video_connections.pop(connection_id, None)
        print(f"Video client {connection_id} disconnected")

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


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global music_processing_task
    
    # Cancel music processing task
    if music_processing_task and not music_processing_task.done():
        music_processing_task.cancel()
        try:
            await music_processing_task
        except asyncio.CancelledError:
            pass
    
    # Close all websocket connections
    for websocket in list(active_connections.values()):
        try:
            await websocket.close()
        except:
            pass
    
    for websocket in list(video_connections.values()):
        try:
            await websocket.close()
        except:
            pass
    
    print("Application shutdown complete")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # 👈 important
    uvicorn.run("main:app", host="0.0.0.0", port=port)