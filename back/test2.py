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
from typing import List, Dict, Tuple

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

class VideoAnalyzer:
    def __init__(self):
        self.places_model = self._load_places_model()
        self.classes = self._load_categories()
        self.prev_gray = None
        
    def _load_categories(self, filename='categories_places365.txt'):
        with open(filename) as f:
            return [line.strip().split(' ')[0][3:] for line in f if line.strip()]
    
    def _load_places_model(self):
        model = models.resnet18(num_classes=365)
        checkpoint = torch.load('resnet18_places365.pth.tar', map_location='cpu')
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
    
    def analyze_frame(self, frame):
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
        motion = self._detect_motion(gray, self.prev_gray)
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
    
    def _detect_motion(self, current_gray, prev_gray):
        if prev_gray is None: return 0
        diff = cv2.absdiff(current_gray, prev_gray)
        return np.mean(diff) / 255.0  # Normalize to 0-1 range
    
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

class MusicPromptGenerator:
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
        self.last_instruments = []
        self.last_genre = ""
        self.last_mood = ""
        
    def generate_prompts(self, analysis: Dict) -> Tuple[List[types.WeightedPrompt], Dict]:
        """Generate music prompts and config updates based on video analysis"""
        prompts = []
        config_updates = {}
        
        # 1. Determine genre using scene analysis
        scene_text = ', '.join([s[0].lower() for s in analysis['top_scenes']])
        scene_emb = model_st.encode(scene_text, convert_to_tensor=True)
        
        # Get top 3 genres with similarity scores
        genre_scores = util.cos_sim(scene_emb, genre_embs)[0]
        top_genre_indices = torch.topk(genre_scores, 3).indices
        top_genres = [(GENRES[i], genre_scores[i].item()) for i in top_genre_indices]
        
        # Smooth genre transition
        current_genre = self._get_best_match(top_genres, self.last_genre)
        self.last_genre = current_genre
        prompts.append(types.WeightedPrompt(text=current_genre, weight=1.0))
        
        # 2. Determine mood using scene analysis and motion
        mood_scores = util.cos_sim(scene_emb, mood_embs)[0]
        
        # Adjust mood based on motion
        motion = analysis['motion']
        if motion > 0.7:
            mood_scores += 0.3 * torch.tensor([1 if m in ["Danceable", "Upbeat", "Fat Beats"] else 0 for m in MOODS]).to(mood_scores.device)
        elif motion < 0.3:
            mood_scores += 0.3 * torch.tensor([1 if m in ["Chill", "Ambient", "Subdued Melody"] else 0 for m in MOODS]).to(mood_scores.device)
        
        top_mood_indices = torch.topk(mood_scores, 2).indices
        top_moods = [(MOODS[i], mood_scores[i].item()) for i in top_mood_indices]
        
        # Smooth mood transition
        current_mood = self._get_best_match(top_moods, self.last_mood)
        self.last_mood = current_mood
        prompts.append(types.WeightedPrompt(text=current_mood, weight=0.8))
        
        # 3. Determine instruments using objects and scene
        object_names = [obj['name'].lower() for obj in analysis['objects']]
        instrument_candidates = []
        
        # Get instruments from detected objects
        for obj in object_names:
            for key, instrument in self.object_to_instrument.items():
                if key in obj:
                    instrument_candidates.append(instrument)
        
        # If no objects detected, use scene-based instruments
        if not instrument_candidates:
            scene_instruments = self._get_instruments_for_scene(scene_text)
            instrument_candidates.extend(scene_instruments)
        
        # Get top 3 instruments with similarity scores
        if instrument_candidates:
            instrument_text = " ".join(instrument_candidates)
            instrument_emb = model_st.encode(instrument_text, convert_to_tensor=True)
            inst_scores = util.cos_sim(instrument_emb, instrument_embs)[0]
            top_inst_indices = torch.topk(inst_scores, min(3, len(INSTRUMENTS))).indices
            top_instruments = [(INSTRUMENTS[i], inst_scores[i].item()) for i in top_inst_indices]
            
            # Smooth instrument transition
            current_instruments = self._get_best_instruments(top_instruments, self.last_instruments)
            self.last_instruments = current_instruments
            
            # Add instruments with decreasing weights
            for i, (inst, score) in enumerate(current_instruments):
                weight = 0.7 - (i * 0.15)  # First instrument gets highest weight
                prompts.append(types.WeightedPrompt(text=inst, weight=weight))
        
        # 4. Add environmental context
        prompts.append(types.WeightedPrompt(text=f"{analysis['brightness']} lighting", weight=0.4))
        prompts.append(types.WeightedPrompt(text=f"{analysis['weather']} weather", weight=0.4))
        prompts.append(types.WeightedPrompt(text=f"{analysis['time_of_day']} time", weight=0.4))
        
        # 5. Update config based on motion and mood
        if motion > 0.7:
            config_updates['bpm'] = 140 + int(20 * motion)
            config_updates['density'] = min(0.9, 0.6 + motion * 0.3)
        elif motion > 0.3:
            config_updates['bpm'] = 100 + int(40 * motion)
            config_updates['density'] = 0.5 + motion * 0.2
        else:
            config_updates['bpm'] = 60 + int(40 * motion)
            config_updates['density'] = 0.3 + motion * 0.2
        
        # Adjust brightness based on lighting
        if analysis['brightness'] == "dark":
            config_updates['brightness'] = 0.3
        elif analysis['brightness'] == "dim":
            config_updates['brightness'] = 0.5
        elif analysis['brightness'] == "medium":
            config_updates['brightness'] = 0.7
        else:
            config_updates['brightness'] = 0.9
        
        return prompts, config_updates
    
    def _get_best_match(self, candidates: List[Tuple[str, float]], last_value: str) -> str:
        """Select best match with consideration of previous value"""
        if not last_value:
            return candidates[0][0]
        
        # Check if last value is in top candidates
        for candidate, score in candidates:
            if candidate == last_value:
                return candidate
                
        # Otherwise return top candidate
        return candidates[0][0]
    
    def _get_best_instruments(self, candidates: List[Tuple[str, float]], last_instruments: List[str]) -> List[Tuple[str, float]]:
        """Select best instruments with consideration of previous instruments"""
        if not last_instruments:
            return candidates[:3]
            
        # Try to keep at least one instrument from previous set
        result = []
        remaining_candidates = candidates.copy()
        
        for last_inst in last_instruments:
            for i, (inst, score) in enumerate(remaining_candidates):
                if inst == last_inst:
                    result.append((inst, score))
                    del remaining_candidates[i]
                    break
            if len(result) >= 2:
                break
                
        # Add remaining top candidates
        result.extend(remaining_candidates[:3 - len(result)])
        return result[:3]
    
    def _get_instruments_for_scene(self, scene_text: str) -> List[str]:
        """Fallback instrument selection based on scene"""
        scene_emb = model_st.encode(scene_text, convert_to_tensor=True)
        inst_scores = util.cos_sim(scene_emb, instrument_embs)[0]
        top_inst_indices = torch.topk(inst_scores, 5).indices
        return [INSTRUMENTS[i] for i in top_inst_indices]

class PromptSmoother:
    def __init__(self):
        self.current_prompts = []
        self.target_prompts = []
        self.transition_start = 0
        self.transition_duration = 8.0  # 8 second transition
    
    def update_target(self, new_prompts: List[types.WeightedPrompt], current_time: float):
        """Set new target prompts for smooth transition"""
        self.target_prompts = new_prompts
        self.transition_start = current_time
    
    def get_current_prompts(self, current_time: float) -> List[types.WeightedPrompt]:
        """Get current prompts with smooth transition"""
        if not self.target_prompts or current_time >= self.transition_start + self.transition_duration:
            self.current_prompts = self.target_prompts
            return self.current_prompts
            
        progress = (current_time - self.transition_start) / self.transition_duration
        blended_prompts = []
        
        # Create a dictionary of current prompts for easy lookup
        current_dict = {p.text: p for p in self.current_prompts}
        target_dict = {p.text: p for p in self.target_prompts}
        
        # Get all unique prompts from both sets
        all_prompts = set(current_dict.keys()).union(set(target_dict.keys()))

        
        
        for prompt_text in all_prompts:
            current_p = current_dict.get(prompt_text)
            target_p = target_dict.get(prompt_text)
            
            if current_p and target_p:
                # Both exist - interpolate weight
                new_weight = current_p.weight * (1 - progress) + target_p.weight * progress
                blended_prompts.append(types.WeightedPrompt(text=prompt_text, weight=new_weight))
            elif current_p:
                # Fading out
                new_weight = current_p.weight * (1 - progress)
                if new_weight > 0.1:  # Don't bother with very small weights
                    blended_prompts.append(types.WeightedPrompt(text=prompt_text, weight=new_weight))
            elif target_p:
                # Fading in
                new_weight = target_p.weight * progress
                if new_weight > 0.1:
                    blended_prompts.append(types.WeightedPrompt(text=prompt_text, weight=new_weight))
        
        # Sort by weight descending
        blended_prompts.sort(key=lambda x: -x.weight)
        return blended_prompts

async def main():
    # Initialize components
    api_key = os.environ.get("LYRIA_API_KEY") or input("Enter API Key: ").strip()
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    video_analyzer = VideoAnalyzer()
    prompt_generator = MusicPromptGenerator()
    prompt_smoother = PromptSmoother()
    p = pyaudio.PyAudio()
    
    # Open video capture
    cap = cv2.VideoCapture('http://192.168.2.106:8080/video')
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    async with client.aio.live.music.connect(model=MODEL) as session:
        print("Connected to music session.")
        
        # Initial configuration
        config = types.LiveMusicGenerationConfig(
            bpm=120,
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
            """Display analysis results on frame"""
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
            
            cv2.imshow("Live Analyzer", frame)
        
        last_update_time = 0
        current_config = config
        
        async def video_processing_loop():
            nonlocal last_update_time, current_config
            print("Press 'q' in the video window to quit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze frame
                analysis = video_analyzer.analyze_frame(frame)
                current_time = time.time()
                
                # Update prompts every 3 seconds or when significant change detected
                if current_time - last_update_time > 3.0:
                    # Generate new prompts and config updates
                    new_prompts, config_updates = prompt_generator.generate_prompts(analysis)
                    
                    # Update configuration if needed
                    if config_updates:
                        for key, value in config_updates.items():
                            setattr(current_config, key, value)
                        await session.set_music_generation_config(config=current_config)
                    
                    # Start smooth transition to new prompts
                    prompt_smoother.update_target(new_prompts, current_time)
                    last_update_time = current_time
                
                # Get current prompts (with smooth transition)
                current_prompts = prompt_smoother.get_current_prompts(current_time)
                if current_prompts:
                    for i, prompt in enumerate(current_prompts, 1):
                        print(f"{i}. {prompt.text} (weight: {prompt.weight:.2f})")
                    await session.set_weighted_prompts(prompts=current_prompts)
                
                # Display analysis
                display_analysis(analysis, frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                await asyncio.sleep(0.1)
            
            cap.release()
            cv2.destroyAllWindows()
        
        # Run both tasks concurrently
        await asyncio.gather(
            receive_audio(),
            video_processing_loop()
        )
    
    p.terminate()

if __name__ == "__main__":
    asyncio.run(main())