# main.py
from fastapi import FastAPI, UploadFile, File, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
from dotenv import load_dotenv
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
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from firebase_admin import auth, initialize_app, credentials
import firebase_admin


app = FastAPI()
security = HTTPBearer()
load_dotenv()

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

# Replace the credential loading with:
firebase_config = {
    "type": "service_account",
    "project_id": os.getenv('FIREBASE_PROJECT_ID'),
    "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
    "private_key": os.getenv('FIREBASE_PRIVATE_KEY'),
    "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
    "client_id": os.getenv('FIREBASE_CLIENT_ID'),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_CERT_URL')
}

cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred)

# Custom logging handler to send logs to WebSocke
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    


@app.post("/verify-token")
async def verify_token(current_user: dict = Depends(get_current_user)):
    return {"status": "authenticated", "uid": current_user['uid']}

from final import start_client_video_processing, update_frame_from_client

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    firebase_config = {
        'apiKey': os.getenv('FIREBASE_API_KEY'),
        'authDomain': os.getenv('FIREBASE_AUTH_DOMAIN'),
        'projectId': os.getenv('FIREBASE_PROJECT_ID'),
        'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET'),
        'messagingSenderId': os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
        'appId': os.getenv('FIREBASE_APP_ID')
    }
    return templates.TemplateResponse("index.html", {"request": request, "firebase_config":json.dumps(firebase_config)})



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
    try:
        data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        token = json.loads(data).get('token')
        if not token:
            await websocket.close(code=1008)
            return
        
        # Verify token
        decoded_token = auth.verify_id_token(token)
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
    except Exception as e:
        await websocket.close(code=1008)
        return

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        token = json.loads(data).get('token')
        if not token:
            await websocket.close(code=1008)
            return
        
        # Verify token
        decoded_token = auth.verify_id_token(token)
        connection_id = str(uuid.uuid4())
        video_connections[connection_id] = websocket
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
    except Exception as e:
        await websocket.close(code=1008)
        return    

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