# main.py
from fastapi import FastAPI, UploadFile, File, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import uuid
import asyncio
from pathlib import Path
import cv2
import time
from typing import Dict
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocketDisconnect
import uvicorn
import json
import asyncio
import threading
import queue
import time
import logging

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
log_connections: Dict[str, WebSocket] = {}

# Custom logging handler to send logs to WebSocket
class WebSocketLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_queue = queue.Queue()
    
    def emit(self, record):
        try:
            msg = self.format(record)
            if not self.log_queue.full():
                self.log_queue.put_nowait(msg)
        except:
            pass
    
    def get_messages(self):
        messages = []
        while not self.log_queue.empty():
            try:
                messages.append(self.log_queue.get_nowait())
            except:
                break
        return messages

# Create the WebSocket log handler
websocket_handler = WebSocketLogHandler()
websocket_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

async def process_log_queue():
    """Process log queue and broadcast to WebSocket clients"""
    while True:
        try:
            messages = websocket_handler.get_messages()
            for message in messages:
                await broadcast_log(message)
        except Exception as e:
            # Use basic print here to avoid recursion
            print(f"Error in log queue processing: {e}")
        await asyncio.sleep(0.1)

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    log_connections[connection_id] = websocket
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle ping/pong if needed
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_text("ping")
                except:
                    break
            except WebSocketDisconnect:
                break
    except:
        pass
    finally:
        log_connections.pop(connection_id, None)

async def broadcast_log(message: str):
    """Safely broadcast log message to all connected WebSocket clients"""
    if not log_connections:
        return
    
    try:
        dead_connections = []
        for connection_id, websocket in log_connections.items():
            try:
                await websocket.send_text(message)
            except:
                dead_connections.append(connection_id)
        
        # Clean up dead connections
        for conn_id in dead_connections:
            log_connections.pop(conn_id, None)
            
    except Exception as e:
        # Use basic print to avoid recursion
        print(f"Error broadcasting log: {e}")

@app.on_event("startup")
async def startup_event():
    # Start the log queue processor
    asyncio.create_task(process_log_queue())
    print("Application started - WebSocket logging enabled")

from final import start_client_video_processing, update_frame_from_client

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start-live")
async def handle_start_live(request: Request):
    global music_processing_task, stop_processing_flag
    
    # Get user settings from request body
    try:
        body = await request.json()
        user_bpm = int(body.get('bpm', 120))
        user_scale = str(body.get('scale', 'C_MAJOR_A_MINOR'))
        user_genre = str(body.get('genre', 'Electronic')).strip()
        print(f"USER SELECTED: BPM={user_bpm}, Scale={user_scale}, Genre={user_genre}")
    except Exception as e:
        logging.error(f"Error parsing user settings: {e}")
        user_bpm = 120
        user_scale = 'C_MAJOR_A_MINOR'
        user_genre = 'Electronic'
    
    # Stop any existing processing first
    stop_processing_flag = True
    if music_processing_task and not music_processing_task.done():
        music_processing_task.cancel()
        try:
            await music_processing_task
        except asyncio.CancelledError:
            pass
    
    # Reset flag and start new processing with user settings
    stop_processing_flag = False
    music_processing_task = asyncio.create_task(
        start_client_video_processing(broadcast_audio, user_bpm, user_scale, user_genre)
    )
    print(f"Live processing started with BPM: {user_bpm}, Scale: {user_scale}, Genre: {user_genre}")
    return {"message": f"Live processing started....."}

@app.post("/stop-live")
async def handle_stop_live():
    global music_processing_task, stop_processing_flag
    
    # Set stop flag
    stop_processing_flag = True
    
    # Cancel existing task if running
    if music_processing_task and not music_processing_task.done():
        music_processing_task.cancel()
        try:
            await music_processing_task
        except asyncio.CancelledError:
            pass
    
    music_processing_task = None
    stop_processing_flag = False
    
    print("Live processing stopped")
    return {"message": "Live processing stopped"}

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    active_connections[connection_id] = websocket
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                try:
                    await websocket.send_text("ping")
                except:
                    break
            except WebSocketDisconnect:
                break
            except Exception as e:
                logging.error(f"WebSocket receive error: {e}")
                break
    except Exception as e:
        logging.error(f"WebSocket connection error: {e}")
    finally:
        active_connections.pop(connection_id, None)
        print(f"Audio client {connection_id} disconnected")

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
                logging.error(f"JSON decode error: {e}")
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
            except Exception as e:
                logging.error(f"Video processing error: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))
                
    except Exception as e:
        logging.error(f"Video WebSocket error: {e}")
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
            logging.error(f"Error broadcasting to {connection_id}: {e}")
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
    port = int(os.environ.get("PORT", 8080))  
    uvicorn.run("main:app", host="0.0.0.0", port=port)