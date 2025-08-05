# final.py
import torch
import torchvision.models as models
import torch.nn.functional as F
import cv2
import numpy as np
import os
import time
import asyncio
from datetime import datetime
from itertools import zip_longest
from torchvision import transforms
from sklearn.cluster import KMeans
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, util
from google import genai
from google.genai import types
from dotenv import load_dotenv
import hashlib
from typing import List, Dict, Tuple, Optional, Callable, Awaitable, Set
from PIL import Image
import glob
import argparse
import json
from dataclasses import dataclass
from enum import Enum

current_frame = None
frame_lock = asyncio.Lock()
frame_timestamp = 0
no_frame_count = 0

async def update_frame_from_client(image_bytes: bytes):
    """Update the current frame from client"""
    global current_frame, frame_timestamp, no_frame_count
    
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            async with frame_lock:
                current_frame = frame
                frame_timestamp = time.time()
                no_frame_count = 0
            print(f"Frame updated: {frame.shape}")
        else:
            print("Failed to decode frame from client")
            
    except Exception as e:
        print(f"Error updating frame from client: {e}")

async def start_client_video_processing(broadcast_func: Callable[[bytes], Awaitable[None]]):
    """Process video frames received from client and generate adaptive music"""
    global current_frame, frame_timestamp, no_frame_count
    
    api_key = os.environ.get("LYRIA_API_KEY") or input("Enter API Key: ").strip()
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    media_analyzer = MediaAnalyzer()
    prompt_generator = EnhancedMusicGenerator()
    
    print("Starting client video processing...")
    
    try:
        async with client.aio.live.music.connect(model=MODEL) as session:
            print("Connected to music session. Waiting for video frames from client...")
            
            # Initial configuration
            config = types.LiveMusicGenerationConfig(
                bpm=120,
                scale=types.Scale.C_MAJOR_A_MINOR,
                brightness=0.7,
                density=0.6,
                guidance=6.0
            )
            
            # Set initial configuration
            await session.set_music_generation_config(config=config)
            print("Initial configuration set")

            # Wait for first frame and collect initial analysis data
            base_prompts = None
            print("Waiting for video frames to start analysis...")
            
            # Wait for first frame with timeout
            max_wait_time = 30  # seconds
            wait_start = time.time()
            
            while current_frame is None:
                if time.time() - wait_start > max_wait_time:
                    print("Timeout waiting for first video frame")
                    return {"status": "error", "message": "No video frames received"}
                
                print("Waiting for first frame from client...")
                await asyncio.sleep(1)
            
            print("First frame received! Starting music generation...")
            
            # Collect initial analysis data and generate base prompts + dynamic pools
            while not prompt_generator.base_prompts_generated:
                async with frame_lock:
                    if current_frame is not None:
                        frame = current_frame.copy()
                    else:
                        print("No frame available, waiting...")
                        await asyncio.sleep(0.1)
                        continue
                
                analysis = media_analyzer.analyze_media(frame, is_video=True)
                temp_prompts, config_updates = await prompt_generator.generate_prompts(
                    analysis, is_video=True)
                
                if prompt_generator.base_prompts_generated:
                    base_prompts = prompt_generator.base_prompts
                    # Set initial base prompts and apply config updates
                    await session.set_weighted_prompts(prompts=base_prompts)
                    
                    # Apply initial config updates
                    if config_updates:
                        for attr, value in config_updates.items():
                            setattr(config, attr, value)
                        await session.set_music_generation_config(config=config)
                    
                    print(f"Base prompts set: {len(base_prompts)} prompts")
                    break
                
                await asyncio.sleep(0.1)

            # Start playback after base prompts are set
            await session.play()
            print("Playback started")
            
            async def receive_audio():
                """Receive and broadcast audio from the music session"""
                chunks_count = 0
                try:
                    async for message in session.receive():
                        chunks_count += 1
                        if chunks_count == 1:
                            print("First audio chunk received, buffering...")
                            await asyncio.sleep(BUFFER_SECONDS)
                        
                        if message.server_content and message.server_content.audio_chunks:
                            audio_data = message.server_content.audio_chunks[0].data
                            await broadcast_func(audio_data)
                        elif message.filtered_prompt:
                            print("Filtered prompt:", message.filtered_prompt)
                        await asyncio.sleep(0)
                except Exception as e:
                    print(f"Error in audio receiver: {str(e)}")
            
            last_update_time = 0
            current_config = config
            audio_task = asyncio.create_task(receive_audio())
            last_frame_time = time.time()
            
            try:
                while True:
                    current_time = time.time()
                    
                    # Check if we're receiving frames
                    async with frame_lock:
                        if current_frame is not None:
                            frame = current_frame.copy()
                            last_frame_time = current_time
                            no_frame_count = 0
                        else:
                            no_frame_count += 1
                            # If no frames for too long, continue with last known frame
                            if current_time - last_frame_time > 5.0:
                                print("No frames received for 5 seconds, continuing with last analysis...")
                                if no_frame_count > 50:  # Reset occasionally
                                    no_frame_count = 0
                            await asyncio.sleep(0.1)
                            continue
                    
                    # Analyze current frame
                    analysis = media_analyzer.analyze_media(frame, is_video=True)
                    
                    # Update prompts and config during live processing
                    if current_time - last_update_time > 4.0 and analysis.get('motion', 0.0) > 0.05:
                        # Generate combined prompts (base + dynamic)
                        all_prompts, config_updates = await prompt_generator.generate_prompts(
                            analysis, is_video=True)
                        
                        # Update configuration if needed
                        config_changed = False
                        if config_updates:
                            for key, value in config_updates.items():
                                if hasattr(current_config, key):
                                    setattr(current_config, key, value)
                                    config_changed = True
                            
                            if config_changed:
                                await session.set_music_generation_config(config=current_config)
                                print("Updated config:", config_updates)
                        
                        # Update prompts with combined base + dynamic
                        print("\n=== CURRENT PROMPTS ===")
                        print("Base Prompts:")
                        for i, prompt in enumerate(base_prompts, 1):
                            print(f"{i}. [BASE] {prompt.text} (weight: {prompt.weight:.2f})")
                        print("\nDynamic Prompts:")
                        for i, prompt in enumerate(prompt_generator.current_dynamic_prompts, 1):
                            print(f"{i}. [DYNAMIC] {prompt.text} (weight: {prompt.weight:.2f})")
                        print("=" * 30)
                        
                        await session.set_weighted_prompts(prompts=all_prompts)
                        last_update_time = current_time
                    
                    # Display analysis (optional - you might want to send this back to client)
                    status = prompt_generator.get_prompt_status()
                    print(f"Motion: {analysis.get('motion', 0):.3f}, "
                          f"Scene: {analysis['top_scenes'][0][0]}, "
                          f"Objects: {len(analysis['objects'])}, "
                          f"Active Prompts: {status.get('num_current_dynamic', 0)}")
                    
                    await asyncio.sleep(0.1)
                    
            finally:
                audio_task.cancel()
                try:
                    await audio_task
                except asyncio.CancelledError:
                    pass
    
    except Exception as e:
        print(f"Error in client video processing session: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        print("Client video processing ended")


async def get_processing_stats():
    """Get current processing statistics"""
    global current_frame, frame_timestamp, no_frame_count
    
    async with frame_lock:
        return {
            "has_frame": current_frame is not None,
            "frame_shape": current_frame.shape if current_frame is not None else None,
            "last_frame_time": frame_timestamp,
            "no_frame_count": no_frame_count,
            "time_since_last_frame": time.time() - frame_timestamp if frame_timestamp > 0 else 0
        }
# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load environment variables
load_dotenv()

# Initialize models globally with GPU if available
model_st = SentenceTransformer('all-MiniLM-L6-v2').to(device)
yolo_model = YOLO('yolov8n.pt').to(device)

# Music configuration
BUFFER_SECONDS = 1
CHUNK = 4200
FORMAT = 'int16'
CHANNELS = 2
MODEL = 'models/lyria-realtime-exp'
OUTPUT_RATE = 48000

# Music parameters database (kept for fallback compatibility)
INSTRUMENTS = [
    "303 Acid Bass", "808 Hip Hop Beat", "Accordion", "Alto Saxophone", "Bagpipes",
    "Balalaika Ensemble", "Banjo", "Bass Clarinet", "Bongos", "Boomy Bass",
    "Bouzouki", "Buchla Synths", "Cello", "Charango", "Clavichord",
    "Conga Drums", "Didgeridoo", "Dirty Synths", "Djembe", "Drumline",
    "Dulcimer", "Fiddle", "Flamenco Guitar", "Funk Drums", "Glockenspiel",    
    "Guitar", "Hang Drum", "Harmonica", "Harp", "Harpsichord",
    "Hurdy-gurdy", "Kalimba", "Koto", "Lyre", "Mandolin",
    "Maracas", "Marimba", "Mbira", "Mellotron", "Metallic Twang",
    "Moog Oscillations", "Ocarina", "Persian Tar", "Pipa", "Precision Bass",
    "Ragtime Piano", "Rhodes Piano", "Shamisen", "Shredding Guitar", "Sitar",
    "Slide Guitar", "Smooth Pianos", "Spacey Synths", "Steel Drum", "Synth Pads",
    "Tabla", "TR-909 Drum Machine", "Trumpet", "Tuba", "Vibraphone",
    "Viola Ensemble", "Warm Acoustic Guitar", "Woodwinds"
]

GENRES = [
    "Acid Jazz", "Afrobeat", "Alternative Country", "Baroque", "Bengal Baul",
    "Bhangra", "Bluegrass", "Blues Rock", "Bossa Nova", "Breakbeat",
    "Celtic Folk", "Chillout", "Chiptune", "Classic Rock", "Contemporary R&B",
    "Cumbia", "Deep House", "Disco Funk", "Drum & Bass", "Dubstep",
    "EDM", "Electro Swing", "Funk Metal", "G-funk", "Garage Rock",
    "Glitch Hop", "Grime", "Hyperpop", "Indian Classical", "Indie Electronic",
    "Indie Folk", "Indie Pop", "Irish Folk", "Jam Band", "Jamaican Dub",
    "Jazz Fusion", "Latin Jazz", "Lo-Fi Hip Hop", "Marching Band", "Merengue",
    "New Jack Swing", "Minimal Techno", "Moombahton", "Neo-Soul", "Orchestral Score",
    "Piano Ballad", "Polka", "Post-Punk", "60s Psychedelic Rock", "Psytrance",
    "R&B", "Reggae", "Reggaeton", "Renaissance Music", "Salsa",
    "Shoegaze", "Ska", "Surf Rock", "Synthpop", "Techno",
    "Trance", "Trap Beat", "Trip Hop", "Vaporwave", "Witch house"
]

MOODS = [
    "Acoustic Instruments", "Bright Tones", "Chill", "Crunchy Distortion",
    "Danceable", "Dreamy", "Echo", "Emotional", "Ethereal Ambience",
    "Experimental", "Fat Beats", "Funky", "Glitchy Effects", "Huge Drop",
    "Live Performance", "Lo-fi", "Ominous Drone", "Psychedelic", "Rich Orchestration",
    "Saturated Tones", "Subdued Melody", "Sustained Chords", "Swirling Phasers",
    "Tight Groove", "Unsettling", "Upbeat", "Virtuoso", "Weird Noises"
]

# Precompute embeddings on GPU if available (kept for fallback)
instrument_embs = model_st.encode(INSTRUMENTS, convert_to_tensor=True).to(device)
genre_embs = model_st.encode(GENRES, convert_to_tensor=True).to(device)
mood_embs = model_st.encode(MOODS, convert_to_tensor=True).to(device)


class PromptCategory(Enum):
    RHYTHMIC = "rhythmic"
    BASS = "bass"
    MELODIC = "melodic"
    PERCUSSIVE = "percussive"
    ATMOSPHERIC = "atmospheric"
    TRANSITION = "transition"
    TEXTURAL = "textural"
    HARMONIC = "harmonic"
    DRUMS = "drums" 

@dataclass
class DynamicPrompt:
    text: str
    category: PromptCategory
    base_weight: float
    triggers: List[str]  # Conditions that activate this prompt
    intensity_range: Tuple[float, float]  # Min/max intensity multipliers
    max_duration: Optional[float] = None  # For time-limited prompts
    cooldown: float = 0.0  # Minimum time between activations

@dataclass
class ActivePrompt:
    prompt: DynamicPrompt
    current_weight: float
    activation_time: float
    target_weight: float
    fade_duration: float = 2.0

class MediaAnalyzer:
    def __init__(self):
        self.places_model = self._load_places_model().to(device)
        self.classes = self._load_categories()
        self.prev_gray = None
        # Initialize depth estimation model
        self.depth_model = self._load_depth_model().to(device)
        self.depth_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),  # Depth model expects 256x256
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for depth model
        ])
        
    def _load_depth_model(self):
        # Using a lightweight depth estimation model
        model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        model.eval()
        return model
    
    def _load_categories(self, filename='./models/categories_places365.txt'):
        with open(filename) as f:
            return [line.strip().split(' ')[0][3:] for line in f if line.strip()]
    
    def _load_places_model(self):
        model = models.resnet18(num_classes=365)
        checkpoint = torch.load('./models/resnet18_places365.pth.tar', map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    def preprocess_image(self, img):
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        return preprocess(img).unsqueeze(0).to(device)
    
    def analyze_depth(self, frame):
        """Estimate depth and return depth characteristics"""
        input_tensor = self.depth_transform(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            depth_pred = self.depth_model(input_tensor)
        
        # Convert to numpy and normalize
        depth_map = depth_pred.squeeze().cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        # Calculate depth characteristics
        avg_depth = np.mean(depth_map)
        depth_variance = np.var(depth_map)
        
        # Classify depth profile
        if depth_variance < 0.01:
            depth_profile = "flat"
        elif avg_depth < 0.3:
            if depth_variance > 0.1:
                depth_profile = "shallow_with_detail"
            else:
                depth_profile = "shallow"
        elif avg_depth > 0.7:
            if depth_variance > 0.1:
                depth_profile = "deep_with_detail"
            else:
                depth_profile = "deep"
        else:
            if depth_variance > 0.15:
                depth_profile = "medium_with_detail"
            else:
                depth_profile = "medium"
        
        # Calculate foreground/background ratio
        fg_ratio = np.mean(depth_map < 0.5)
        
        return {
            'depth_map': depth_map,
            'depth_profile': depth_profile,
            'avg_depth': float(avg_depth),
            'depth_variance': float(depth_variance),
            'fg_ratio': float(fg_ratio)
        }
    
    def analyze_media(self, media_path: str, is_video: bool = False):
        if is_video:
            return self.analyze_video_frame(media_path)
        else:
            return self.analyze_photo(media_path)
    
    def analyze_video_frame(self, frame):
        frame_resized = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # Scene analysis
        input_tensor = self.preprocess_image(frame_resized)
        with torch.no_grad():
            probs = F.softmax(self.places_model(input_tensor), 1)[0].cpu()
        top_probs, top_idxs = torch.topk(probs, 3)
        top_scenes = [(self.classes[i], float(p)) for i, p in zip(top_idxs, top_probs)]
        
        # Environmental analysis
        brightness = self._get_brightness(frame_resized)
        weather = self._simulate_weather(frame_resized)
        time_of_day = self._get_time_of_day()
        motion = self._detect_motion(gray, self.prev_gray) if self.prev_gray is not None else 0.0
        self.prev_gray = gray
        objects = self._detect_objects(frame_resized)
        colors = self._get_dominant_colors(frame_resized)
        depth = self.analyze_depth(frame_resized)  # New depth analysis
        
        return {
            'top_scenes': top_scenes,
            'brightness': brightness,
            'weather': weather,
            'time_of_day': time_of_day,
            'motion': motion,
            'objects': objects,
            'colors': colors,
            'depth': depth,
            'frame_resized': frame_resized
        }
    
    def analyze_photo(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        img_resized = cv2.resize(img, (640, 480))
        
        input_tensor = self.preprocess_image(img_resized)
        with torch.no_grad():
            probs = F.softmax(self.places_model(input_tensor), 1)[0].cpu()
        top_probs, top_idxs = torch.topk(probs, 3)
        top_scenes = [(self.classes[i], float(p)) for i, p in zip(top_idxs, top_probs)]
        
        brightness = self._get_brightness(img_resized)
        weather = self._simulate_weather(img_resized)
        time_of_day = self._get_time_of_day_from_image(img_resized)
        objects = self._detect_objects(img_resized)
        colors = self._get_dominant_colors(img_resized)
        depth = self.analyze_depth(img_resized)  # New depth analysis
        
        return {
            'top_scenes': top_scenes,
            'brightness': brightness,
            'weather': weather,
            'time_of_day': time_of_day,
            'motion': 0.0,
            'objects': objects,
            'colors': colors,
            'depth': depth,
            'frame_resized': img_resized
        }
    
    def _get_brightness(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        if brightness < 50: return "dark"
        elif brightness < 100: return "dim"
        elif brightness < 180: return "medium"
        return "bright"
    
    def _simulate_weather(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = np.mean(hsv[:, :, 0])
        brightness = np.mean(hsv[:, :, 2])
        saturation = np.mean(hsv[:, :, 1])
        
        if brightness < 60 and saturation > 40: return "rainy"
        elif hue > 90 and brightness > 160: return "sunny"
        elif 50 < hue < 90 and saturation < 60: return "cloudy"
        elif brightness > 200 and saturation < 30: return "fog"
        return "clear"
    
    def _get_time_of_day(self):
        hour = datetime.now().hour
        if 5 <= hour < 12: return "morning"
        elif 12 <= hour < 17: return "afternoon"
        elif 17 <= hour < 20: return "evening"
        return "night"
    
    def _get_time_of_day_from_image(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        saturation = np.mean(hsv[:, :, 1])
        
        if saturation < 40 and 50 < brightness < 150:
            return "night"
            
        if brightness > 180:
            return "day"
        elif brightness > 120:
            return "evening"
        elif brightness > 70:
            return "dusk/dawn"
        return "night"
    
    def _detect_motion(self, current_gray, prev_gray):
        if prev_gray is None: return 0
        diff = cv2.absdiff(current_gray, prev_gray)
        return np.mean(diff) / 255.0
    
    def _detect_objects(self, frame):
        # Move frame to GPU if available
        frame_gpu = torch.from_numpy(frame).to(device) if torch.cuda.is_available() else frame
        results = yolo_model(frame_gpu, verbose=False)[0]
        names = yolo_model.names
        detected_objects = []
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if float(conf) > 0.5:
                detected_objects.append({
                    'name': names[int(cls)],
                    'confidence': float(conf)
                })
        return detected_objects
    
    def _get_dominant_colors(self, frame, k=3):
        data = frame.reshape((-1, 3)).astype(np.float32)
        kmeans = KMeans(n_clusters=k, n_init=1, random_state=42)
        kmeans.fit(data)
        colors = kmeans.cluster_centers_.astype(int)
        color_names = []
        for c in colors:
            if c[0] > 200 and c[1] > 200 and c[2] > 200:
                color_names.append("white")
            elif c[0] < 50 and c[1] < 50 and c[2] < 50:
                color_names.append("black")
            elif c[2] > max(c[0], c[1]) + 50:
                color_names.append("red")
            elif c[1] > max(c[0], c[2]) + 50:
                color_names.append("green")
            elif c[0] > max(c[1], c[2]) + 50:
                color_names.append("blue")
            elif c[0] > 150 and c[1] > 150 and c[2] < 100:
                color_names.append("yellow")
            else:
                color_names.append("neutral")
        return color_names[:3]

class EnhancedMusicGenerator:
    def __init__(self):
        # Initialize AI client for prompt generation
        self.ai_client = None
        self.base_prompts = []
        self.dynamic_prompt_pools = {}  # Category -> List[DynamicPrompt]
        self.base_prompts_generated = False
        
        # Live processing state
        self.active_prompts = {}  # prompt.text -> ActivePrompt
        self.prompt_history = {}  # prompt.text -> last_activation_time
        self.scene_context_history = []
        self.motion_window = []
        self.brightness_history = []
        
        # Analysis aggregation for first 5 seconds
        self.initial_analysis_data = []
        self.analysis_start_time = None
        
        # Musical balance constraints - Updated with DRUMS category
        self.max_concurrent_prompts = {
            PromptCategory.RHYTHMIC: 2,
            PromptCategory.BASS: 1,
            PromptCategory.MELODIC: 1,
            PromptCategory.PERCUSSIVE: 1,
            PromptCategory.ATMOSPHERIC: 1,
            PromptCategory.TRANSITION: 1,
            PromptCategory.TEXTURAL: 1,
            PromptCategory.HARMONIC: 1,
            PromptCategory.DRUMS: 3  # Allow up to 3 concurrent drum elements
        }
        
        # Current dynamic prompts for monitoring
        self.current_dynamic_prompts = []

    def initialize_ai_client(self):
        """Initialize the AI client for prompt generation"""
        try:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("No AI API key found. Set GEMINI_API_KEY environment variable.")
            
            import google.generativeai as genai_client
            genai_client.configure(api_key=api_key)
            
            model_names_to_try = ['gemini-2.0-flash-exp', 'gemini-1.5-flash']
            
            for model_name in model_names_to_try:
                try:
                    self.ai_client = genai_client.GenerativeModel(model_name)
                    test_response = self.ai_client.generate_content("Hello")
                    print(f"AI client initialized successfully with model: {model_name}")
                    return
                except Exception as model_error:
                    print(f"Model {model_name} failed: {model_error}")
                    continue
            
            raise Exception("No working Gemini model found")
            
        except Exception as e:
            print(f"Failed to initialize AI client: {e}")
            self.ai_client = None

    def collect_initial_analysis(self, analysis: Dict) -> bool:
        """Collect analysis data for the first 5 seconds"""
        if self.analysis_start_time is None:
            self.analysis_start_time = time.time()
            print("Starting initial 5-second analysis collection...")
        
        current_time = time.time()
        analysis_with_time = {
            **analysis,
            'timestamp': current_time - self.analysis_start_time
        }
        self.initial_analysis_data.append(analysis_with_time)
        
        if current_time - self.analysis_start_time >= 5.0:
            print(f"Collected {len(self.initial_analysis_data)} analysis frames over 5 seconds")
            return True
        
        return False

    def aggregate_analysis_data(self) -> Dict:
        """Aggregate the collected 5-second analysis data"""
        if not self.initial_analysis_data:
            return {}
        
        # Aggregate scenes
        scene_counts = {}
        for analysis in self.initial_analysis_data:
            for scene, confidence in analysis['top_scenes']:
                if scene not in scene_counts:
                    scene_counts[scene] = []
                scene_counts[scene].append(confidence)
        
        avg_scene_confidence = {scene: np.mean(confidences) 
                               for scene, confidences in scene_counts.items()}
        top_consistent_scenes = sorted(avg_scene_confidence.items(), 
                                     key=lambda x: -x[1])[:5]
        
        # Aggregate objects
        object_counts = {}
        for analysis in self.initial_analysis_data:
            for obj in analysis['objects']:
                obj_name = obj['name']
                if obj_name not in object_counts:
                    object_counts[obj_name] = []
                object_counts[obj_name].append(obj['confidence'])
        
        avg_object_confidence = {obj: np.mean(confidences) 
                               for obj, confidences in object_counts.items()}
        top_consistent_objects = sorted(avg_object_confidence.items(), 
                                      key=lambda x: -x[1])[:8]
        
        # Environmental data
        brightness_modes = [a['brightness'] for a in self.initial_analysis_data]
        weather_modes = [a['weather'] for a in self.initial_analysis_data]
        time_modes = [a['time_of_day'] for a in self.initial_analysis_data]
        
        most_common_brightness = max(set(brightness_modes), key=brightness_modes.count)
        most_common_weather = max(set(weather_modes), key=weather_modes.count)
        most_common_time = max(set(time_modes), key=time_modes.count)
        
        # Motion analysis
        motion_values = [a.get('motion', 0) for a in self.initial_analysis_data]
        avg_motion = np.mean(motion_values)
        motion_variance = np.var(motion_values)
        
        # Color analysis
        all_colors = []
        for analysis in self.initial_analysis_data:
            all_colors.extend(analysis['colors'])
        color_counts = {color: all_colors.count(color) for color in set(all_colors)}
        dominant_colors = sorted(color_counts.items(), key=lambda x: -x[1])[:5]
        
        # Depth analysis
        depth_profiles = [a['depth']['depth_profile'] for a in self.initial_analysis_data 
                         if 'depth' in a and 'depth_profile' in a['depth']]
        most_common_depth = max(set(depth_profiles), key=depth_profiles.count) if depth_profiles else "medium"
        
        avg_depth_values = [a['depth']['avg_depth'] for a in self.initial_analysis_data 
                           if 'depth' in a and 'avg_depth' in a['depth']]
        avg_depth = np.mean(avg_depth_values) if avg_depth_values else 0.5
        
        return {
            'scenes': top_consistent_scenes,
            'objects': top_consistent_objects,
            'brightness': most_common_brightness,
            'weather': most_common_weather,
            'time_of_day': most_common_time,
            'avg_motion': avg_motion,
            'motion_variance': motion_variance,
            'dominant_colors': dominant_colors,
            'depth_profile': most_common_depth,
            'avg_depth': avg_depth,
            'total_frames': len(self.initial_analysis_data)
        }

    async def generate_base_config_and_prompts(self, aggregated_data: Dict) -> Tuple[types.LiveMusicGenerationConfig, List[types.WeightedPrompt]]:
        """Generate initial configuration, base prompts, and dynamic prompt pools"""
        if not self.ai_client:
            self.initialize_ai_client()
            if not self.ai_client:
                return self._fallback_base_config(aggregated_data), self._fallback_base_prompts(aggregated_data)
        
        ai_prompt = f"""
you are electronic dance genre dj , Based on this comprehensive video scene analysis, generate a complete adaptive music system including:
1. Base music configuration
2. Core foundational prompts (base prompts that stay constant)
3. Dynamic prompt pools for live adaptation

Scene Analysis:
- Main Scenes: {', '.join([f"{scene} ({conf:.2f})" for scene, conf in aggregated_data['scenes'][:3]])}
- Key Objects: {', '.join([f"{obj} ({conf:.2f})" for obj, conf in aggregated_data['objects'][:5]])}
- Environment: {aggregated_data['brightness']} lighting, {aggregated_data['weather']}, {aggregated_data['time_of_day']}
- Motion Level: {aggregated_data['avg_motion']:.2f} (0=static, 1=dynamic)
- Motion Variance: {aggregated_data['motion_variance']:.3f}
- Depth Profile: {aggregated_data['depth_profile']}
- Colors: {', '.join([color for color, _ in aggregated_data['dominant_colors'][:3]])}

chose scale_name from this-exact same keyword - no twisted name or new name: C_MAJOR_A_MINOR, D_FLAT_MAJOR_B_FLAT_MINOR, D_MAJOR_B_MINOR, E_FLAT_MAJOR_C_MINOR, E_MAJOR_D_FLAT_MINOR, F_MAJOR_D_MINOR, G_FLAT_MAJOR_E_FLAT_MINOR, G_MAJOR_E_MINOR, A_FLAT_MAJOR_F_MINOR, A_MAJOR_G_FLAT_MINOR, B_FLAT_MAJOR_G_MINOR, B_MAJOR_A_FLAT_MINOR

motherfucker dont change my scale name, just use the same name as it is

IMPORTANT: Use ONLY these triggers that actually work in the video analysis system:
- Motion triggers: "high_motion", "medium_motion", "calm_moment", "motion_change", "motion_spike", "energy_spike"
- Scene triggers: "scene_change", "dramatic_shift"
- Object triggers: "action_scene", "human_presence"
- Depth triggers: "deep_scene", "spatial_depth"
- Drum triggers: "drum_moment", "beat_emphasis"

Generate JSON in this exact format:
{{
    "config": {{
        "bpm": <60-200>,
        "scale": E_MAJOR_D_FLAT_MINOR,
        "density": <0.0-1.0>,
        "brightness": <0.0-2.0>,
        "guidance": <1.0-6.0>
    }},
    "base_prompts": [
        {{"text": "<core genre/main instrument/primary mood>", "weight": <1.0-5.0>}}
    ],
    "dynamic_pools": {{
        "rhythmic": [
            {{
                "text": "<drum element like kick, snare, hi-hat>",
                "base_weight": <1.0-4.0>,
                "triggers": ["high_motion", "energy_spike", "drum_moment"],
                "intensity_range": [0.5, 2.0]
            }}
        ],
        "bass": [
            {{
                "text": "<bass element like sub bass, bass hit, bass drop>",
                "base_weight": <1.0-4.0>,
                "triggers": ["motion_spike", "scene_change", "energy_spike"],
                "intensity_range": [0.8, 2.5]
            }}
        ],
        "melodic": [
            {{
                "text": "<melodic element like lead synth, pad, arpeggio>",
                "base_weight": <1.0-3.5>,
                "triggers": ["calm_moment", "deep_scene", "spatial_depth"],
                "intensity_range": [0.5, 2.0]
            }}
        ],
        "percussive": [
            {{
                "text": "<percussion like shaker, clap, rim shot>",
                "base_weight": <0.8-3.0>,
                "triggers": ["medium_motion", "beat_emphasis", "action_scene"],
                "intensity_range": [0.3, 1.8]
            }}
        ],
        "atmospheric": [
            {{
                "text": "<atmosphere like reverb, ambient texture, pad>",
                "base_weight": <1.0-3.0>,
                "triggers": ["deep_scene", "calm_moment", "spatial_depth"],
                "intensity_range": [0.5, 1.5]
            }}
        ],
        "transition": [
            {{
                "text": "<transition like build-up, drop, sweep>",
                "base_weight": <2.0-4.0>,
                "triggers": ["scene_change", "motion_change", "dramatic_shift"],
                "intensity_range": [1.0, 3.0],
                "max_duration": 4.0,
                "cooldown": 8.0
            }}
        ],
        "textural": [
            {{
                "text": "<texture like filter sweep, phaser, chorus>",
                "base_weight": <0.5-2.0>,
                "triggers": ["medium_motion", "spatial_depth"],
                "intensity_range": [0.2, 1.2]
            }}
        ],
        "harmonic": [
            {{
                "text": "<harmony like chord stab, sustained note, harmonic layer>",
                "base_weight": <1.0-3.0>,
                "triggers": ["calm_moment", "deep_scene", "spatial_depth"],
                "intensity_range": [0.5, 2.0]
            }}
        ],
        "drums": [
            {{
                "text": "<drum kit element like kick drum, snare hit, cymbal crash, hi-hat pattern>",
                "base_weight": <1.5-4.0>,
                "triggers": ["high_motion", "action_scene", "energy_spike", "drum_moment", "beat_emphasis"],
                "intensity_range": [0.8, 3.5]
            }}
        ]
    }}
}}

Guidelines:
- Base prompts (4-6): Core elements that define the musical style and stay constant
- Dynamic pools: Each category should have 3-6 contextually relevant options
- DRUMS category: Focus on expressive drum kit elements that respond to video action and energy
- Drums should trigger on high motion, scene changes, and moments needing rhythmic emphasis
- All prompts should work together harmoniously within the chosen genre/style
- Consider the scene type when choosing instruments (e.g., electronic for urban, acoustic for nature)
"""

        try:
            response = self.ai_client.generate_content(ai_prompt)
            response_text = response.text.strip()
            
            # Clean up response text
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
                
            result = json.loads(response_text)
            print("AI Response for base config and prompts generated successfully")
            
            # Process configuration
            config_data = result.get('config', {})
            config = types.LiveMusicGenerationConfig(
                bpm=min(max(60, config_data['bpm']), 200),
                scale=getattr(types.Scale, config_data['scale']),
                density=min(max(0.0, config_data['density']), 1.0),
                brightness=min(max(0.0, config_data['brightness']), 2.0),
                guidance=min(max(1.0, config_data['guidance']), 6.0)
            )
            
            # Process base prompts
            base_prompts = []
            for prompt_data in result.get('base_prompts', []):
                text = prompt_data['text']
                weight = float(prompt_data['weight'])
                base_prompts.append(types.WeightedPrompt(text=text, weight=weight))
                print(f"Added base prompt: {text} (weight: {weight:.2f})")
            
            # Process dynamic prompt pools
            dynamic_pools_data = result.get('dynamic_pools', {})
            self.dynamic_prompt_pools = {}
            
            for category_name, prompts_data in dynamic_pools_data.items():
                try:
                    category = PromptCategory(category_name)
                    self.dynamic_prompt_pools[category] = []
                    
                    for prompt_data in prompts_data:
                        dynamic_prompt = DynamicPrompt(
                            text=prompt_data['text'],
                            category=category,
                            base_weight=float(prompt_data['base_weight']),
                            triggers=prompt_data['triggers'],
                            intensity_range=(float(prompt_data['intensity_range'][0]), 
                                           float(prompt_data['intensity_range'][1])),
                            max_duration=prompt_data.get('max_duration'),
                            cooldown=prompt_data.get('cooldown', 0.0)
                        )
                        self.dynamic_prompt_pools[category].append(dynamic_prompt)
                        print(f"Added {category_name} prompt: {dynamic_prompt.text}")
                
                except ValueError as e:
                    print(f"Unknown category {category_name}, skipping: {e}")
                    continue
            
            total_dynamic_prompts = sum(len(prompts) for prompts in self.dynamic_prompt_pools.values())
            print(f"Successfully generated: {len(base_prompts)} base prompts, {total_dynamic_prompts} dynamic prompts across {len(self.dynamic_prompt_pools)} categories")
            
            return config, base_prompts
                
        except Exception as e:
            print(f"Error generating config and prompts: {e}")
            return self._fallback_base_config(aggregated_data), self._fallback_base_prompts(aggregated_data)

    def _analyze_context_triggers(self, analysis: Dict) -> Set[str]:
        """Analyze current context and return active triggers - Cleaned version with only working triggers"""
        triggers = set()
        
        # Motion-based triggers
        motion = analysis.get('motion', 0.0)
        if motion > 0.20:
            triggers.add('high_motion')
            triggers.add('action_scene')
            triggers.add('energy_spike')
        elif motion > 0.15:
            triggers.add('medium_motion')
        else:
            triggers.add('calm_moment')
        
        # Motion change detection
        if len(self.motion_window) > 5:
            recent_motion = np.array(self.motion_window[-5:])
            motion_change = np.std(recent_motion)
            if motion_change > 0.2:
                triggers.add('motion_change')
                triggers.add('energy_spike')
            if motion > 0.8 and any(m < 0.2 for m in recent_motion[-3:]):
                triggers.add('motion_spike')
                triggers.add('drum_moment')
        
        # Scene-based triggers
        depth_info = analysis.get('depth', {})
        depth_profile = depth_info.get('depth_profile', 'medium')
        if 'deep' in depth_profile:
            triggers.add('deep_scene')
            triggers.add('spatial_depth')
        
        # Scene change detection
        current_scenes = [scene for scene, _ in analysis['top_scenes'][:2]]
        if len(self.scene_context_history) > 3:
            prev_scenes = self.scene_context_history[-2]
            if not any(scene in prev_scenes for scene in current_scenes):
                triggers.add('scene_change')
                triggers.add('dramatic_shift')
                triggers.add('drum_moment')
        
        # Object-based triggers
        objects = analysis.get('objects', [])
        if any(obj['name'] == 'person' for obj in objects):
            triggers.add('human_presence')
        
        # Check for action-related objects
        action_objects = ['car', 'truck', 'motorcycle', 'bicycle', 'sports ball', 'skateboard']
        if any(obj['name'] in action_objects for obj in objects):
            triggers.add('action_scene')
            triggers.add('energy_spike')
        
        # Beat emphasis trigger (simplified)
        current_time = time.time()
        if hasattr(self, 'last_beat_time'):
            time_since_beat = current_time - self.last_beat_time
            if time_since_beat > 3.0:  # Every 3 seconds
                triggers.add('beat_emphasis')
                self.last_beat_time = current_time
        else:
            self.last_beat_time = current_time
            triggers.add('beat_emphasis')
        
        # Store scene context for next comparison
        self.scene_context_history.append(current_scenes)
        if len(self.scene_context_history) > 5:
            self.scene_context_history.pop(0)
        
        return triggers

    def _calculate_dynamic_intensity(self, analysis: Dict, prompt: DynamicPrompt, triggers: Set[str]) -> float:
        """Calculate intensity multiplier for a dynamic prompt based on current context"""
        intensity = 1.0
        
        # Base intensity from trigger matching
        matching_triggers = set(prompt.triggers).intersection(triggers)
        trigger_strength = len(matching_triggers) / len(prompt.triggers)
        
        # Motion influence
        motion = analysis.get('motion', 0.0)
        if prompt.category in [PromptCategory.RHYTHMIC, PromptCategory.BASS, PromptCategory.DRUMS]:
            intensity *= (0.5 + motion * 1.5)
        elif prompt.category == PromptCategory.ATMOSPHERIC:
            intensity *= (1.2 - motion * 0.4)  # More atmospheric when less motion
        
        # Special intensity boost for drums during high-energy moments
        if prompt.category == PromptCategory.DRUMS:
            if motion > 0.7:
                intensity *= 1.3  # Extra boost for drums during high motion
            if 'action_scene' in triggers or 'energy_spike' in triggers:
                intensity *= 1.2  # Additional boost for action scenes
        
        # Depth influence
        depth_info = analysis.get('depth', {})
        avg_depth = depth_info.get('avg_depth', 0.5)
        if prompt.category == PromptCategory.ATMOSPHERIC:
            intensity *= (0.8 + avg_depth * 0.8)
        elif prompt.category == PromptCategory.DRUMS and avg_depth < 0.3:
            intensity *= 1.1  # Boost drums for shallow/close-up scenes
        
        # Apply trigger strength
        intensity *= (0.3 + trigger_strength * 1.4)
        
        # Clamp to prompt's intensity range
        min_intensity, max_intensity = prompt.intensity_range
        intensity = max(min_intensity, min(max_intensity, intensity))
        
        return intensity

    def _select_dynamic_prompts(self, analysis: Dict) -> List[types.WeightedPrompt]:
        """Select and weight dynamic prompts based on current context"""
        if not self.dynamic_prompt_pools:
            return []
        
        current_time = time.time()
        triggers = self._analyze_context_triggers(analysis)
        selected_prompts = []
        
        # Update motion window
        self.motion_window.append(analysis.get('motion', 0.0))
        if len(self.motion_window) > 20:
            self.motion_window.pop(0)
        
        print(f"Active triggers: {triggers}")
        
        # Process each category
        for category, prompts in self.dynamic_prompt_pools.items():
            category_selections = []
            
            for prompt in prompts:
                # Check cooldown
                last_activation = self.prompt_history.get(prompt.text, 0)
                if current_time - last_activation < prompt.cooldown:
                    continue
                
                # Check if triggers match
                matching_triggers = set(prompt.triggers).intersection(triggers)
                if not matching_triggers:
                    continue
                
                # Calculate relevance score
                trigger_score = len(matching_triggers) / len(prompt.triggers)
                intensity = self._calculate_dynamic_intensity(analysis, prompt, triggers)
                relevance_score = trigger_score * intensity
                
                category_selections.append((prompt, relevance_score, intensity))
            
            # Sort by relevance and select best ones for this category
            category_selections.sort(key=lambda x: x[1], reverse=True)
            max_concurrent = self.max_concurrent_prompts.get(category, 2)
            
            for prompt, relevance_score, intensity in category_selections[:max_concurrent]:
                if relevance_score > 0.3:  # Minimum relevance threshold
                    final_weight = prompt.base_weight * intensity
                    selected_prompts.append(types.WeightedPrompt(
                        text=prompt.text,
                        weight=final_weight
                    ))
                    
                    # Update prompt history
                    self.prompt_history[prompt.text] = current_time
                    
                    print(f"Selected {category.value}: {prompt.text} (weight: {final_weight:.2f}, relevance: {relevance_score:.2f})")
        
        # Store current dynamic prompts for display
        self.current_dynamic_prompts = selected_prompts
        
        return selected_prompts

    async def generate_prompts(self, analysis: Dict, is_video: bool) -> Tuple[List[types.WeightedPrompt], Dict]:
        """Main method to generate prompts - handles both base prompt generation and dynamic updates"""
        config_updates = {}
        
        # Phase 1: Collect initial data and generate base prompts + dynamic pools
        if not self.base_prompts_generated:
            collection_complete = self.collect_initial_analysis(analysis)
            
            if collection_complete:
                print("5-second analysis complete. Generating base configuration, prompts, and dynamic pools...")
                aggregated_data = self.aggregate_analysis_data()
                
                # Generate config, base prompts, and dynamic pools
                initial_config, self.base_prompts = await self.generate_base_config_and_prompts(aggregated_data)
                self.base_prompts_generated = True
                
                # Set initial configuration
                config_updates = {
                    'bpm': initial_config.bpm,
                    'scale': initial_config.scale,
                    'density': initial_config.density,
                    'brightness': initial_config.brightness,
                    'guidance': initial_config.guidance
                }
                
                print(f"System initialized: {len(self.base_prompts)} base prompts, {sum(len(p) for p in self.dynamic_prompt_pools.values())} dynamic prompts")
            
            # During collection phase, return minimal prompts
            if not self.base_prompts_generated:
                temp_prompts = [
                    types.WeightedPrompt(text="Ambient", weight=0.8),
                    types.WeightedPrompt(text="Chill", weight=0.6)
                ]
                return temp_prompts, config_updates
        
        # Phase 2: Live processing with intelligent dynamic prompt selection
        all_prompts = []
        
        # Add stable base prompts
        all_prompts.extend(self.base_prompts)
        
        # Add intelligently selected dynamic prompts
        dynamic_prompts = self._select_dynamic_prompts(analysis)
        all_prompts.extend(dynamic_prompts)
        
        # Update configuration based on current conditions
        if is_video:
            motion = analysis.get('motion', 0)
            brightness = analysis.get('brightness', 'medium')
            
            # Smooth motion-based density adjustment
            if len(self.motion_window) > 5:
                smoothed_motion = np.mean(self.motion_window[-5:])
                base_density = 0.5
                motion_adjustment = smoothed_motion * 0.3
                config_updates['density'] = max(0.2, min(0.9, base_density + motion_adjustment))
            
            # Brightness-based adjustments
            if brightness == "dark":
                config_updates['brightness'] = 1.4
            elif brightness == "bright":
                config_updates['brightness'] = 1.8
            else:
                config_updates['brightness'] = 1.6
        
        return all_prompts, config_updates

    def get_prompt_status(self) -> Dict:
        """Get current status of prompt generation system"""
        active_categories = {cat.value: len(prompts) for cat, prompts in self.dynamic_prompt_pools.items()}
        
        return {
            'base_prompts_generated': self.base_prompts_generated,
            'num_base_prompts': len(self.base_prompts),
            'dynamic_pool_sizes': active_categories,
            'total_dynamic_prompts': sum(len(prompts) for prompts in self.dynamic_prompt_pools.values()),
            'num_current_dynamic': len(self.current_dynamic_prompts),
            'analysis_frames_collected': len(self.initial_analysis_data),
            'system_ready': self.base_prompts_generated and bool(self.dynamic_prompt_pools)
        }

    def _fallback_base_config(self, aggregated_data: Dict) -> types.LiveMusicGenerationConfig:
        """Fallback configuration if AI generation fails"""
        avg_motion = aggregated_data.get('avg_motion', 0.5)
        
        return types.LiveMusicGenerationConfig(
            bpm=int(100 + avg_motion * 60),  # 100-160 BPM based on motion
            scale=types.Scale.C_MAJOR_A_MINOR,
            density=0.4 + avg_motion * 0.3,
            brightness=1.6,
            guidance=5.0
        )

    def _fallback_base_prompts(self, aggregated_data: Dict) -> List[types.WeightedPrompt]:
        """Fallback base prompts if AI generation fails - Updated with cleaned triggers"""
        prompts = [
            types.WeightedPrompt(text="Ambient", weight=2.0),
            types.WeightedPrompt(text="Chill", weight=1.5),
            types.WeightedPrompt(text="Guitar", weight=1.8)
        ]
        
        # Add simple dynamic pool as fallback with cleaned triggers
        self.dynamic_prompt_pools = {
            PromptCategory.RHYTHMIC: [
                DynamicPrompt("kick", PromptCategory.RHYTHMIC, 2.0, ["high_motion", "energy_spike"], (1.0, 3.0)),
                DynamicPrompt("snare", PromptCategory.RHYTHMIC, 1.5, ["medium_motion", "beat_emphasis"], (0.8, 2.5))
            ],
            PromptCategory.ATMOSPHERIC: [
                DynamicPrompt("reverb", PromptCategory.ATMOSPHERIC, 1.8, ["deep_scene", "calm_moment"], (0.8, 2.0))
            ],
            PromptCategory.DRUMS: [
                DynamicPrompt("kick drum", PromptCategory.DRUMS, 2.5, ["high_motion", "action_scene", "beat_emphasis"], (1.0, 3.5)),
                DynamicPrompt("snare hit", PromptCategory.DRUMS, 2.0, ["energy_spike", "drum_moment"], (0.8, 3.0)),
                DynamicPrompt("hi-hat", PromptCategory.DRUMS, 1.8, ["medium_motion", "beat_emphasis"], (0.6, 2.5)),
                DynamicPrompt("cymbal crash", PromptCategory.DRUMS, 3.0, ["scene_change", "dramatic_shift"], (1.5, 4.0), max_duration=3.0, cooldown=5.0)
            ]
        }
        
        return prompts

async def live_camera_processing(broadcast_func: Callable[[bytes], Awaitable[None]]):
    """Process live camera feed and generate adaptive music"""
    api_key = os.environ.get("LYRIA_API_KEY") or input("Enter API Key: ").strip()
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    media_analyzer = MediaAnalyzer()
    prompt_generator = EnhancedMusicGenerator()
    
    # Camera setup with timeout and retry
    cap = None
    for _ in range(3):
        try:
            cap = cv2.VideoCapture('http://192.168.2.106:8080/video')
            if cap.isOpened():
                break
        except Exception as e:
            print(f"Camera open attempt failed: {str(e)}")
            await asyncio.sleep(1)
    
    if not cap or not cap.isOpened():
        print("Error: Could not open camera after multiple attempts")
        return

    try:
        async with client.aio.live.music.connect(model=MODEL) as session:
            print("Connected to music session. Starting initial 5-second analysis...")
            
            # Initial configuration
            config = types.LiveMusicGenerationConfig(
                bpm=120,
                scale=types.Scale.C_MAJOR_A_MINOR,
                brightness=0.7,
                density=0.6,
                guidance=6.0
            )
            
            # Set initial configuration
            await session.set_music_generation_config(config=config)
            print("Initial configuration set")

            # Collect initial analysis data and generate base prompts + dynamic pools
            base_prompts = None
            while not prompt_generator.base_prompts_generated:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                analysis = media_analyzer.analyze_media(frame, is_video=True)
                temp_prompts, config_updates = await prompt_generator.generate_prompts(
                    analysis, is_video=True)
                
                if prompt_generator.base_prompts_generated:
                    base_prompts = prompt_generator.base_prompts
                    # Set initial base prompts and apply config updates
                    await session.set_weighted_prompts(prompts=base_prompts)
                    
                    # Apply initial config updates
                    if config_updates:
                        for attr, value in config_updates.items():
                            setattr(config, attr, value)
                        await session.set_music_generation_config(config=config)
                    
                    print(f"Base prompts set: {len(base_prompts)} prompts")
                    break
                
                await asyncio.sleep(0.1)

            # Start playback after base prompts are set
            await session.play()
            print("Playback started")
            
            async def receive_audio():
                chunks_count = 0
                try:
                    async for message in session.receive():
                        chunks_count += 1
                        if chunks_count == 1:
                            print("First audio chunk received, buffering...")
                            await asyncio.sleep(BUFFER_SECONDS)
                        
                        if message.server_content and message.server_content.audio_chunks:
                            audio_data = message.server_content.audio_chunks[0].data
                            await broadcast_func(audio_data)
                        elif message.filtered_prompt:
                            print("Filtered prompt:", message.filtered_prompt)
                        await asyncio.sleep(0)
                except Exception as e:
                    print(f"Error in audio receiver: {str(e)}")
            
            last_update_time = 0
            current_config = config
            audio_task = asyncio.create_task(receive_audio())
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Camera frame read failed")
                        break
                    
                    analysis = media_analyzer.analyze_media(frame, is_video=True)
                    current_time = time.time()
                    
                    # Update prompts and config during live processing
                    if current_time - last_update_time > 4.0 and analysis.get('motion', 0.0) > 0.1:
                        # Generate combined prompts (base + dynamic)
                        all_prompts, config_updates = await prompt_generator.generate_prompts(
                            analysis, is_video=True)
                        
                        # Update configuration if needed
                        config_changed = False
                        if config_updates:
                            for key, value in config_updates.items():
                                if hasattr(current_config, key):
                                    setattr(current_config, key, value)
                                    config_changed = True
                            
                            if config_changed:
                                await session.set_music_generation_config(config=current_config)
                                print("Updated config:", config_updates)
                        
                        # Update prompts with combined base + dynamic
                        print("\n=== CURRENT PROMPTS ===")
                        print("Base Prompts:")
                        for i, prompt in enumerate(base_prompts, 1):
                            print(f"{i}. [BASE] {prompt.text} (weight: {prompt.weight:.2f})")
                        print("\nDynamic Prompts:")
                        for i, prompt in enumerate(prompt_generator.current_dynamic_prompts, 1):
                            print(f"{i}. [DYNAMIC] {prompt.text} (weight: {prompt.weight:.2f})")
                        print("=" * 30)
                        
                        await session.set_weighted_prompts(prompts=all_prompts)
                        last_update_time = current_time
                    
                    # Display analysis
                    status = prompt_generator.get_prompt_status()
                    display_analysis(analysis, analysis['frame_resized'], status)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    await asyncio.sleep(0.1)
                    
            finally:
                audio_task.cancel()
                try:
                    await audio_task
                except asyncio.CancelledError:
                    pass
    
    except Exception as e:
        print(f"Error in main session: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up")

def display_analysis(analysis, frame, status):
    """Enhanced display function showing dynamic prompt system status"""
    object_names = ', '.join([obj['name'] for obj in analysis['objects']]) if analysis['objects'] else 'None'
    depth_info = analysis.get('depth', {})
    
    info = [
        f"=== SCENE ANALYSIS ===",
        f"Scene 1: {analysis['top_scenes'][0][0]} ({analysis['top_scenes'][0][1]*100:.1f}%)",
        f"Scene 2: {analysis['top_scenes'][1][0]} ({analysis['top_scenes'][1][1]*100:.1f}%)",
        f"Scene 3: {analysis['top_scenes'][2][0]} ({analysis['top_scenes'][2][1]*100:.1f}%)",
        f"Objects: {object_names}",
        f"Lighting: {analysis['brightness']}",
        f"Weather: {analysis['weather']}",
        f"Time: {analysis['time_of_day']}",
        f"Motion: {analysis['motion']:.2f}" if 'motion' in analysis else "Photo (no motion)",
        f"Colors: {' '.join(analysis['colors'])}",
        f"Depth: {depth_info.get('depth_profile', 'N/A')}",
        f"",
        f"=== MUSIC SYSTEM STATUS ===",
        f"Phase: {'COLLECTION' if not status['base_prompts_generated'] else 'LIVE PROCESSING'}",
        f"Base Prompts: {status['num_base_prompts']}",
        f"Dynamic Pools: {sum(status['dynamic_pool_sizes'].values()) if 'dynamic_pool_sizes' in status else 0}",
        f"Active Dynamic: {status.get('num_current_dynamic', 0)}",
        f"System Ready: {'YES' if status.get('system_ready', False) else 'NO'}",
        f"",
        f"=== DYNAMIC CATEGORIES ===",
    ]
    
    # Add dynamic pool status
    if 'dynamic_pool_sizes' in status:
        for category, count in status['dynamic_pool_sizes'].items():
            info.append(f"{category.capitalize()}: {count}")
    
    y = 20
    for line in info:
        if line.startswith("==="):
            color = (0, 255, 255)  # Yellow for headers
        elif line.startswith("Phase:"):
            color = (0, 255, 0) if status['base_prompts_generated'] else (0, 165, 255)  # Green/Orange
        elif line.startswith("System Ready:"):
            color = (0, 255, 0) if status.get('system_ready', False) else (0, 0, 255)  # Green/Red
        else:
            color = (255, 255, 255)  # White for regular info
            
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 20
    
    # Show depth map if available
    if 'depth' in analysis and 'depth_map' in analysis['depth']:
        depth_map = analysis['depth']['depth_map']
        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        depth_map = cv2.resize(depth_map, (frame.shape[1] // 4, frame.shape[0] // 4))
        
        # Overlay depth map on the frame
        frame[10:10+depth_map.shape[0], frame.shape[1]-depth_map.shape[1]-10:frame.shape[1]-10] = depth_map
    
    cv2.imshow("Enhanced Adaptive Music System", frame)
    cv2.waitKey(1)

async def start_live_processing(broadcast_func: Callable[[bytes], Awaitable[None]]):
    """Wrapper for live camera processing"""
    try:
        await live_camera_processing(broadcast_func)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}