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
import pyaudio
from google import genai
from google.genai import types
from dotenv import load_dotenv
import hashlib
from typing import List, Dict, Tuple, Optional
from PIL import Image
import glob
import argparse

# Load environment variables
load_dotenv()

# Initialize models globally
model_st = SentenceTransformer('all-MiniLM-L6-v2')
yolo_model = YOLO('yolov8n.pt')

# Music configuration
BUFFER_SECONDS = 1
CHUNK = 4200
FORMAT = pyaudio.paInt16
CHANNELS = 2
MODEL = 'models/lyria-realtime-exp'
OUTPUT_RATE = 48000

# Music parameters database
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
    "Acoustic Instruments", "Ambient", "Bright Tones", "Chill", "Crunchy Distortion",
    "Danceable", "Dreamy", "Echo", "Emotional", "Ethereal Ambience",
    "Experimental", "Fat Beats", "Funky", "Glitchy Effects", "Huge Drop",
    "Live Performance", "Lo-fi", "Ominous Drone", "Psychedelic", "Rich Orchestration",
    "Saturated Tones", "Subdued Melody", "Sustained Chords", "Swirling Phasers",
    "Tight Groove", "Unsettling", "Upbeat", "Virtuoso", "Weird Noises"
]

# Precompute embeddings
instrument_embs = model_st.encode(INSTRUMENTS, convert_to_tensor=True)
genre_embs = model_st.encode(GENRES, convert_to_tensor=True)
mood_embs = model_st.encode(MOODS, convert_to_tensor=True)

class MediaAnalyzer:
    def __init__(self):
        self.places_model = self._load_places_model()
        self.classes = self._load_categories()
        self.prev_gray = None
        
    def _load_categories(self, filename='./models/categories_places365.txt'):
        with open(filename) as f:
            return [line.strip().split(' ')[0][3:] for line in f if line.strip()]
    
    def _load_places_model(self):
        model = models.resnet18(num_classes=365)
        checkpoint = torch.load('./models/resnet18_places365.pth.tar', map_location='cpu')
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
        return preprocess(img).unsqueeze(0)
    
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
            probs = F.softmax(self.places_model(input_tensor), 1)[0]
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
        
        return {
            'top_scenes': top_scenes,
            'brightness': brightness,
            'weather': weather,
            'time_of_day': time_of_day,
            'motion': motion,
            'objects': objects,
            'colors': colors,
            'frame_resized': frame_resized
        }
    
    def analyze_photo(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        img_resized = cv2.resize(img, (640, 480))
        
        input_tensor = self.preprocess_image(img_resized)
        with torch.no_grad():
            probs = F.softmax(self.places_model(input_tensor), 1)[0]
        top_probs, top_idxs = torch.topk(probs, 3)
        top_scenes = [(self.classes[i], float(p)) for i, p in zip(top_idxs, top_probs)]
        
        brightness = self._get_brightness(img_resized)
        weather = self._simulate_weather(img_resized)
        time_of_day = self._get_time_of_day_from_image(img_resized)
        objects = self._detect_objects(img_resized)
        colors = self._get_dominant_colors(img_resized)
        
        return {
            'top_scenes': top_scenes,
            'brightness': brightness,
            'weather': weather,
            'time_of_day': time_of_day,
            'motion': 0.0,
            'objects': objects,
            'colors': colors,
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
        results = yolo_model(frame, verbose=False)[0]
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
        kmeans = KMeans(n_clusters=k, n_init=1)
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
        self.object_to_instrument = {
            'person': "Vocal Choir",
            'car': "Dirty Synths",
            'tree': "Wind Chimes",
            'dog': "Whistles",
            'cat': "Purring Sounds",
            'computer': "Chiptune",
            'phone': "Electronic Beeps",
            'book': "Page Turning Sounds",
            'chair': "Wooden Percussion",
            'cup': "Glass Harmonica",
            'guitar': "Acoustic Guitar",
            'piano': "Grand Piano",
            'violin': "String Ensemble",
            'clock': "Ticking Sounds"
        }
        self.genre_history = []
        self.mood_history = []
        self.instrument_history = []
        self.motion_window = []
        self.genre_lock = False
        self.target_instrument_weights = {}
        self.last_genre = ""
        self.last_mood = ""
        
    def generate_prompts(self, analysis: Dict, is_video: bool) -> Tuple[List[types.WeightedPrompt], Dict]:
        prompts = []
        config_updates = {}
        
        # 1. Stable Genre Selection
        scene_text = ', '.join([s[0].lower() for s in analysis['top_scenes']])
        scene_emb = model_st.encode(scene_text, convert_to_tensor=True)
        
        if not self.genre_lock or not self.last_genre:
            genre_scores = util.cos_sim(scene_emb, genre_embs)[0]
            
            # History bias
            for i, genre in enumerate(GENRES):
                if genre in self.genre_history[-3:]:
                    genre_scores[i] += 0.2
            
            top_genres = [(GENRES[i], float(s)) for i,s in enumerate(genre_scores)][:3]
            self.last_genre = self._get_best_match(top_genres, self.last_genre)
            self.genre_history.append(self.last_genre)
            if len(self.genre_history) > 5:
                self.genre_history.pop(0)
            
            if self.genre_history.count(self.last_genre) >= 3:
                self.genre_lock = True
        
        prompts.append(types.WeightedPrompt(text=self.last_genre, weight=1.0))
        
        # 2. Adaptive Mood Selection
        mood_scores = util.cos_sim(scene_emb, mood_embs)[0]
        
        if is_video:
            self.motion_window.append(analysis['motion'])
            if len(self.motion_window) > 5:
                self.motion_window.pop(0)
            smoothed_motion = sum(self.motion_window)/len(self.motion_window)
            
            motion_factor = min(1.0, smoothed_motion * 1.5)
            for i, mood in enumerate(MOODS):
                if "Upbeat" in mood or "Danceable" in mood:
                    mood_scores[i] += 0.3 * motion_factor
                elif "Chill" in mood or "Ambient" in mood:
                    mood_scores[i] += 0.3 * (1 - motion_factor)
        
        # Mood history bias
        for i, mood in enumerate(MOODS):
            if mood in self.mood_history[-2:]:
                mood_scores[i] += 0.15
                
        top_moods = [(MOODS[i], float(s)) for i,s in enumerate(mood_scores)][:2]
        self.last_mood = self._get_best_match(top_moods, self.last_mood)
        self.mood_history.append(self.last_mood)
        if len(self.mood_history) > 4:
            self.mood_history.pop(0)
            
        prompts.append(types.WeightedPrompt(text=self.last_mood, weight=0.8))
        
        # 3. Motion-Modulated Instruments
        object_names = [obj['name'].lower() for obj in analysis['objects']]
        instrument_candidates = []
        
        for obj in object_names:
            for key, instrument in self.object_to_instrument.items():
                if key in obj:
                    instrument_candidates.append(instrument)
        
        if not instrument_candidates:
            instrument_candidates = self._get_instruments_for_scene(scene_text)
        
        # Calculate new target weights
        new_target_weights = {}
        if instrument_candidates:
            instrument_text = " ".join(instrument_candidates)
            instrument_emb = model_st.encode(instrument_text, convert_to_tensor=True)
            inst_scores = util.cos_sim(instrument_emb, instrument_embs)[0]
            
            base_weights = {INSTRUMENTS[i]: float(s)*0.7 for i,s in enumerate(inst_scores)}
            
            motion_factor = smoothed_motion if is_video else 0.5
            for inst in base_weights:
                if "Drum" in inst or "Beat" in inst:
                    new_target_weights[inst] = base_weights[inst] * (0.8 + motion_factor*0.4)
                elif "Pad" in inst or "String" in inst:
                    new_target_weights[inst] = base_weights[inst] * (0.8 + (1-motion_factor)*0.4)
                else:
                    new_target_weights[inst] = base_weights[inst]
        
        # Smooth weight transitions
        if not self.target_instrument_weights:
            self.target_instrument_weights = new_target_weights
        else:
            for inst in self.target_instrument_weights:
                if inst in new_target_weights:
                    self.target_instrument_weights[inst] = (
                        0.8 * self.target_instrument_weights[inst] + 
                        0.2 * new_target_weights[inst]
                    )
        
        # Add instruments to prompts
        sorted_instruments = sorted(
            self.target_instrument_weights.items(), 
            key=lambda x: -x[1]
        )[:3]
        
        for i, (inst, weight) in enumerate(sorted_instruments):
            final_weight = weight * (1.0 - i*0.15)
            prompts.append(types.WeightedPrompt(text=inst, weight=final_weight))
        
        # 4. Environmental Context
        prompts.append(types.WeightedPrompt(
            text=f"{analysis['brightness']} lighting", 
            weight=0.4 + (smoothed_motion*0.1 if is_video else 0)
        ))
        prompts.append(types.WeightedPrompt(
            text=f"{analysis['weather']} weather", 
            weight=0.4
        ))
        
        # 5. Stable Configuration
        if is_video:
            config_updates['density'] = 0.5 + (smoothed_motion * 0.3)
            if analysis['brightness'] == "dark":
                config_updates['brightness'] = 0.4
            elif analysis['brightness'] == "bright":
                config_updates['brightness'] = 0.8
        else:
            config_updates['density'] = 0.5
        
        return prompts, config_updates
    
    def _get_best_match(self, candidates: List[Tuple[str, float]], last_value: str) -> str:
        if not last_value:
            return candidates[0][0]
        
        for candidate, score in candidates:
            if candidate == last_value:
                return candidate
                
        return candidates[0][0]
    
    def _get_instruments_for_scene(self, scene_text: str) -> List[str]:
        scene_emb = model_st.encode(scene_text, convert_to_tensor=True)
        inst_scores = util.cos_sim(scene_emb, instrument_embs)[0]
        top_inst_indices = torch.topk(inst_scores, 5).indices
        return [INSTRUMENTS[i] for i in top_inst_indices]

class PromptSmoother:
    def __init__(self):
        self.current_prompts = []
        self.target_prompts = []
        self.transition_start = 0
        self.transition_duration = 8.0
    
    def update_target(self, new_prompts: List[types.WeightedPrompt], current_time: float):
        self.target_prompts = new_prompts
        self.transition_start = current_time
    
    def get_current_prompts(self, current_time: float) -> List[types.WeightedPrompt]:
        if not self.target_prompts or current_time >= self.transition_start + self.transition_duration:
            self.current_prompts = self.target_prompts
            return self.current_prompts
            
        progress = (current_time - self.transition_start) / self.transition_duration
        blended_prompts = []
        
        current_dict = {p.text: p for p in self.current_prompts}
        target_dict = {p.text: p for p in self.target_prompts}
        
        all_prompts = set(current_dict.keys()).union(set(target_dict.keys()))
        
        for prompt_text in all_prompts:
            current_p = current_dict.get(prompt_text)
            target_p = target_dict.get(prompt_text)
            
            if current_p and target_p:
                new_weight = current_p.weight * (1 - progress) + target_p.weight * progress
                blended_prompts.append(types.WeightedPrompt(text=prompt_text, weight=new_weight))
            elif current_p:
                new_weight = current_p.weight * (1 - progress)
                if new_weight > 0.1:
                    blended_prompts.append(types.WeightedPrompt(text=prompt_text, weight=new_weight))
            elif target_p:
                new_weight = target_p.weight * progress
                if new_weight > 0.1:
                    blended_prompts.append(types.WeightedPrompt(text=prompt_text, weight=new_weight))
        
        blended_prompts.sort(key=lambda x: -x.weight)
        return blended_prompts

async def process_media(media_path: str, is_video: bool = False):
    api_key = os.environ.get("LYRIA_API_KEY") or input("Enter API Key: ").strip()
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    media_analyzer = MediaAnalyzer()
    prompt_generator = EnhancedMusicGenerator()
    prompt_smoother = PromptSmoother()
    p = pyaudio.PyAudio()
    
    if is_video:
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {media_path}")
            return
    else:
        if not os.path.exists(media_path):
            print(f"Error: Could not find photo at {media_path}")
            return
    
    async with client.aio.live.music.connect(model=MODEL) as session:
        print("Connected to music session.")
        
        config = types.LiveMusicGenerationConfig(
            bpm=120,  # Fixed BPM
            scale=types.Scale.C_MAJOR_A_MINOR,
            brightness=0.7,
            density=0.6,
            guidance=6.0
        )
        await session.set_music_generation_config(config=config)
        await session.play()
        
        async def receive_audio():
            chunks_count = 0
            output_stream = p.open(
                format=FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True, frames_per_buffer=CHUNK)
            try:
                async for message in session.receive():
                    chunks_count += 1
                    if chunks_count == 1:
                        await asyncio.sleep(BUFFER_SECONDS)
                    if message.server_content:
                        audio_data = message.server_content.audio_chunks[0].data
                        output_stream.write(audio_data)
                    elif message.filtered_prompt:
                        print("Filtered prompt:", message.filtered_prompt)
                    await asyncio.sleep(10**-12)
            finally:
                output_stream.close()
        
        def display_analysis(analysis, frame):
            object_names = ', '.join([obj['name'] for obj in analysis['objects']]) if analysis['objects'] else 'None'
            info = [
                f"Scene 1: {analysis['top_scenes'][0][0]} ({analysis['top_scenes'][0][1]*100:.1f}%)",
                f"Scene 2: {analysis['top_scenes'][1][0]} ({analysis['top_scenes'][1][1]*100:.1f}%)",
                f"Scene 3: {analysis['top_scenes'][2][0]} ({analysis['top_scenes'][2][1]*100:.1f}%)",
                f"Objects: {object_names}",
                f"Lighting: {analysis['brightness']}",
                f"Weather: {analysis['weather']}",
                f"Time: {analysis['time_of_day']}",
                f"Motion: {analysis['motion']:.2f}" if is_video else "Photo (no motion)",
                f"Colors: {' '.join(analysis['colors'])}"
            ]
            
            y = 30
            for line in info:
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25
            
            cv2.imshow("Media Analyzer", frame)
            cv2.waitKey(1)
        
        last_update_time = 0
        current_config = config
        
        async def media_processing_loop():
            nonlocal last_update_time, current_config
            if is_video:
                print(f"Processing video: {media_path}")
                print("Press 'q' in the video window to quit.")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    analysis = media_analyzer.analyze_media(frame, is_video=True)
                    current_time = time.time()
                    
                    if current_time - last_update_time > 3.0:
                        new_prompts, config_updates = prompt_generator.generate_prompts(analysis, is_video=True)
                        
                        if config_updates:
                            for key, value in config_updates.items():
                                setattr(current_config, key, value)
                            await session.set_music_generation_config(config=current_config)
                        
                        prompt_smoother.update_target(new_prompts, current_time)
                        last_update_time = current_time
                    
                    current_prompts = prompt_smoother.get_current_prompts(current_time)
                    if current_prompts:
                        for i, prompt in enumerate(current_prompts, 1):
                            print(f"{i}. {prompt.text} (weight: {prompt.weight:.2f})")
                        await session.set_weighted_prompts(prompts=current_prompts)
                    
                    display_analysis(analysis, analysis['frame_resized'])
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    await asyncio.sleep(0.1)
                
                cap.release()
            else:
                print(f"Processing photo: {media_path}")
                
                analysis = media_analyzer.analyze_media(media_path, is_video=False)
                current_time = time.time()
                
                new_prompts, config_updates = prompt_generator.generate_prompts(analysis, is_video=False)
                
                if config_updates:
                    for key, value in config_updates.items():
                        setattr(current_config, key, value)
                    await session.set_music_generation_config(config=current_config)
                
                if new_prompts:
                    for i, prompt in enumerate(new_prompts, 1):
                        print(f"{i}. {prompt.text} (weight: {prompt.weight:.2f})")
                    await session.set_weighted_prompts(prompts=new_prompts)
                
                display_analysis(analysis, analysis['frame_resized'])
                
                while True:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    await asyncio.sleep(0.1)
            
            cv2.destroyAllWindows()
        
        await asyncio.gather(
            receive_audio(),
            media_processing_loop()
        )
    
    p.terminate()

async def batch_process_photos(photo_dir: str):
    photo_paths = glob.glob(os.path.join(photo_dir, "*.jpg")) + glob.glob(os.path.join(photo_dir, "*.png"))
    
    for photo_path in photo_paths:
        print(f"\nProcessing photo: {photo_path}")
        await process_media(photo_path, is_video=False)
        await asyncio.sleep(1)

async def live_camera_processing():
    api_key = os.environ.get("LYRIA_API_KEY") or input("Enter API Key: ").strip()
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    media_analyzer = MediaAnalyzer()
    prompt_generator = EnhancedMusicGenerator()
    prompt_smoother = PromptSmoother()
    p = pyaudio.PyAudio()
    
    # Camera setup with timeout and retry
    cap = None
    for _ in range(3):  # Try 3 times to open camera
        try:
            cap = cv2.VideoCapture('https://192.168.2.106:8080/video')
            if cap.isOpened():
                break
        except Exception as e:
            print(f"Camera open attempt failed: {str(e)}")
            await asyncio.sleep(1)
    
    if not cap or not cap.isOpened():
        print("Error: Could not open camera after multiple attempts")
        p.terminate()
        return
    
    try:
        async with client.aio.live.music.connect(model=MODEL) as session:
            print("Connected to music session. Press 'q' to quit.")
            
            # Initial configuration with verification
            config = types.LiveMusicGenerationConfig(
                bpm=120,
                scale=types.Scale.C_MAJOR_A_MINOR,
                brightness=0.7,
                density=0.6,
                guidance=6.0
            )
            await session.set_music_generation_config(config=config)
            print("Configuration set successfully")
            
            # Start playback with verification
            await session.play()
            print("Playback started successfully")
            
            async def receive_audio():
                try:
                    output_stream = p.open(
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=OUTPUT_RATE,
                        output=True,
                        frames_per_buffer=CHUNK
                    )
                    print("Audio output stream opened successfully")
                    
                    chunks_count = 0
                    async for message in session.receive():
                        chunks_count += 1
                        if chunks_count == 1:
                            print("First audio chunk received, buffering...")
                            await asyncio.sleep(BUFFER_SECONDS)
                        
                        if message.server_content and message.server_content.audio_chunks:
                            audio_data = message.server_content.audio_chunks[0].data
                            output_stream.write(audio_data)
                            print(f"Received audio chunk {chunks_count}")  # Debug output
                        elif message.filtered_prompt:
                            print("Filtered prompt:", message.filtered_prompt)
                        await asyncio.sleep(0)  # Yield control
                except Exception as e:
                    print(f"Error in audio receiver: {str(e)}")
                finally:
                    output_stream.close()
                    print("Audio output stream closed")
            
            last_update_time = 0
            current_config = config
            audio_task = asyncio.create_task(receive_audio())
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Camera frame read failed")
                        break
                    
                    # Analysis and prompt generation
                    analysis = media_analyzer.analyze_media(frame, is_video=True)
                    current_time = time.time()
                    
                    if current_time - last_update_time > 3.0:
                        new_prompts, config_updates = prompt_generator.generate_prompts(
                            analysis, is_video=True)
                        
                        if config_updates:
                            for key, value in config_updates.items():
                                setattr(current_config, key, value)
                            await session.set_music_generation_config(config=current_config)
                            print("Updated config:", config_updates)
                        
                        prompt_smoother.update_target(new_prompts, current_time)
                        last_update_time = current_time
                    
                    current_prompts = prompt_smoother.get_current_prompts(current_time)
                    if current_prompts:
                        print("\nCurrent prompts:")
                        for i, prompt in enumerate(current_prompts, 1):
                            print(f"{i}. {prompt.text} (weight: {prompt.weight:.2f})")
                        await session.set_weighted_prompts(prompts=current_prompts)
                    
                    display_analysis(analysis, analysis['frame_resized'])
                    
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
        p.terminate()
        print("Resources cleaned up")

def display_analysis(analysis, frame):
    object_names = ', '.join([obj['name'] for obj in analysis['objects']]) if analysis['objects'] else 'None'
    info = [
        f"Scene 1: {analysis['top_scenes'][0][0]} ({analysis['top_scenes'][0][1]*100:.1f}%)",
        f"Scene 2: {analysis['top_scenes'][1][0]} ({analysis['top_scenes'][1][1]*100:.1f}%)",
        f"Scene 3: {analysis['top_scenes'][2][0]} ({analysis['top_scenes'][2][1]*100:.1f}%)",
        f"Objects: {object_names}",
        f"Lighting: {analysis['brightness']}",
        f"Weather: {analysis['weather']}",
        f"Time: {analysis['time_of_day']}",
        f"Motion: {analysis['motion']:.2f}",
        f"Colors: {' '.join(analysis['colors'])}"
    ]
    
    y = 30
    for line in info:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25
    
    cv2.imshow("Media Analyzer", frame)
    cv2.waitKey(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music from video or photos")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--photo", type=str, help="Path to single photo file")
    parser.add_argument("--photo-dir", type=str, help="Directory containing photos to process")
    parser.add_argument("--live", action="store_true", help="Use live camera feed")
    
    args = parser.parse_args()
    
    if args.video:
        asyncio.run(process_media(args.video, is_video=True))
    elif args.photo:
        asyncio.run(process_media(args.photo, is_video=False))
    elif args.photo_dir:
        asyncio.run(batch_process_photos(args.photo_dir))
    elif args.live:
        asyncio.run(live_camera_processing())
    else:
        print("Please specify either --video, --photo, --photo-dir, or --live")

# Add these at the end of final.py (don't modify existing code)

async def process_upload(file_path: str, is_video: bool):
    """Wrapper for your existing process_media function"""
    try:
        await process_media(file_path, is_video)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def start_live_processing():
    """Wrapper for live camera processing"""
    try:
        await live_camera_processing()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}