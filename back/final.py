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
from typing import List, Dict, Tuple, Optional, Callable, Awaitable
from PIL import Image
import glob
import argparse
import json

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

# Precompute embeddings on GPU if available
instrument_embs = model_st.encode(INSTRUMENTS, convert_to_tensor=True).to(device)
genre_embs = model_st.encode(GENRES, convert_to_tensor=True).to(device)
mood_embs = model_st.encode(MOODS, convert_to_tensor=True).to(device)

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
        # Initialize AI client for prompt generation
        self.ai_client = None
        self.base_prompts = []
        self.base_prompts_generated = False
        
        # Tracking variables for dynamic prompts
        self.motion_window = []
        self.brightness_history = []
        self.current_dynamic_prompts = []
        
        # Analysis aggregation for first 5 seconds
        self.initial_analysis_data = []
        self.analysis_start_time = None
        
    def initialize_ai_client(self):
        """Initialize the AI client for prompt generation"""
        try:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("No AI API key found. Set GEMINI_API_KEY environment variable.")
            
            import google.generativeai as genai_client
            genai_client.configure(api_key=api_key)
            
            # Try different model names that are currently available
            model_names_to_try = [
                'gemini-1.5-flash',
                'gemini-1.5-pro', 
                'gemini-1.0-pro',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro',
                'models/gemini-1.0-pro'
            ]
            
            for model_name in model_names_to_try:
                try:
                    self.ai_client = genai_client.GenerativeModel(model_name)
                    # Test the model with a simple prompt
                    test_response = self.ai_client.generate_content("Hello")
                    print(f"AI client initialized successfully with model: {model_name}")
                    return
                except Exception as model_error:
                    print(f"Model {model_name} failed: {model_error}")
                    continue
            
            # If all models fail, list available models
            try:
                print("Listing available models:")
                for model in genai_client.list_models():
                    if 'generateContent' in model.supported_generation_methods:
                        print(f"Available model: {model.name}")
            except Exception as list_error:
                print(f"Could not list models: {list_error}")
            
            raise Exception("No working Gemini model found")
            
        except Exception as e:
            print(f"Failed to initialize AI client: {e}")
            self.ai_client = None

    def collect_initial_analysis(self, analysis: Dict) -> bool:
        """
        Collect analysis data for the first 5 seconds
        Returns True when 5 seconds of data is collected
        """
        if self.analysis_start_time is None:
            self.analysis_start_time = time.time()
            print("Starting initial 5-second analysis collection...")
        
        # Store analysis data
        current_time = time.time()
        analysis_with_time = {
            **analysis,
            'timestamp': current_time - self.analysis_start_time
        }
        self.initial_analysis_data.append(analysis_with_time)
        
        # Check if 5 seconds have passed
        if current_time - self.analysis_start_time >= 5.0:
            print(f"Collected {len(self.initial_analysis_data)} analysis frames over 5 seconds")
            return True
        
        return False

    def aggregate_analysis_data(self) -> Dict:
        """
        Aggregate the collected 5-second analysis data into a comprehensive summary
        """
        if not self.initial_analysis_data:
            return {}
        
        # Aggregate scenes with confidence scores
        scene_counts = {}
        for analysis in self.initial_analysis_data:
            for scene, confidence in analysis['top_scenes']:
                if scene not in scene_counts:
                    scene_counts[scene] = []
                scene_counts[scene].append(confidence)
        
        # Get most consistent scenes
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
        
        # Get most consistent objects
        avg_object_confidence = {obj: np.mean(confidences) 
                               for obj, confidences in object_counts.items()}
        top_consistent_objects = sorted(avg_object_confidence.items(), 
                                      key=lambda x: -x[1])[:8]
        
        # Aggregate environmental data
        brightness_modes = [a['brightness'] for a in self.initial_analysis_data]
        weather_modes = [a['weather'] for a in self.initial_analysis_data]
        time_modes = [a['time_of_day'] for a in self.initial_analysis_data]
        
        # Get most common environmental conditions
        most_common_brightness = max(set(brightness_modes), key=brightness_modes.count)
        most_common_weather = max(set(weather_modes), key=weather_modes.count)
        most_common_time = max(set(time_modes), key=time_modes.count)
        
        # Aggregate motion data
        motion_values = [a.get('motion', 0) for a in self.initial_analysis_data]
        avg_motion = np.mean(motion_values)
        motion_variance = np.var(motion_values)
        
        # Aggregate colors
        all_colors = []
        for analysis in self.initial_analysis_data:
            all_colors.extend(analysis['colors'])
        color_counts = {color: all_colors.count(color) for color in set(all_colors)}
        dominant_colors = sorted(color_counts.items(), key=lambda x: -x[1])[:5]
        
        # Aggregate depth information
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

    async def generate_base_prompts_with_ai(self, aggregated_data: Dict) -> List[types.WeightedPrompt]:
        """
        Use AI to generate base prompts from aggregated analysis data
        """
        if not self.ai_client:
            self.initialize_ai_client()
            if not self.ai_client:
                return self._fallback_base_prompts(aggregated_data)
        
        # Create detailed prompt for AI
        ai_prompt = f"""
Based on the following 5-second video analysis data, generate music prompts that would create an appropriate soundtrack. 

Video Analysis Data:
- Main Scenes: {', '.join([f"{scene} ({conf:.2f})" for scene, conf in aggregated_data['scenes'][:3]])}
- Key Objects: {', '.join([f"{obj} ({conf:.2f})" for obj, conf in aggregated_data['objects'][:5]])}
- Lighting: {aggregated_data['brightness']}
- Weather: {aggregated_data['weather']}
- Time of Day: {aggregated_data['time_of_day']}
- Average Motion: {aggregated_data['avg_motion']:.2f} (0=static, 1=very dynamic)
- Motion Variance: {aggregated_data['motion_variance']:.3f}
- Dominant Colors: {', '.join([color for color, count in aggregated_data['dominant_colors'][:3]])}
- Depth Profile: {aggregated_data['depth_profile']}
- Average Depth: {aggregated_data['avg_depth']:.2f}

Available Music Elements:
GENRES: {', '.join(GENRES[:20])}... (and more)
INSTRUMENTS: {', '.join(INSTRUMENTS[:20])}... (and more)  
MOODS: {', '.join(MOODS[:20])}... (and more)

Generate 4-6 base music prompts that would create a soundtrack matching this video content. Each prompt should be:
1. One specific genre, instrument, or mood from the available options
2. Appropriate for the analyzed content
3. Weighted from 0.5 to 1.0 based on relevance

Format your response as JSON:
{{
  "prompts": [
    {{"text": "Jazz Fusion", "weight": 0.9, "reasoning": "Urban scene with moderate motion"}},
    {{"text": "Rhodes Piano", "weight": 0.8, "reasoning": "Matches the indoor sophisticated setting"}},
    ...
  ]
}}
"""

        try:
            # Use the correct Gemini API syntax
            response = self.ai_client.generate_content(ai_prompt)
            response_text = response.text.strip()
            
            # Parse AI response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
                
            ai_result = json.loads(response_text)
            
            base_prompts = []
            for prompt_data in ai_result.get('prompts', []):
                text = prompt_data['text']
                weight = float(prompt_data['weight'])
                
                # Validate that the prompt text exists in our available options
                if (text in GENRES or text in INSTRUMENTS or text in MOODS):
                    base_prompts.append(types.WeightedPrompt(text=text, weight=weight))
                    print(f"AI Base Prompt: {text} (weight: {weight:.2f}) - {prompt_data.get('reasoning', '')}")
            
            if len(base_prompts) >= 3:
                print(f"Successfully generated {len(base_prompts)} base prompts with AI")
                return base_prompts
            else:
                print("AI generated insufficient valid prompts, using fallback")
                return self._fallback_base_prompts(aggregated_data)
                
        except Exception as e:
            print(f"AI prompt generation failed: {e}")
            return self._fallback_base_prompts(aggregated_data)

    def _fallback_base_prompts(self, aggregated_data: Dict) -> List[types.WeightedPrompt]:
        """
        Fallback method to generate base prompts without AI
        """
        print("Using fallback base prompt generation")
        
        base_prompts = []
        
        # Genre based on scene and motion
        primary_scene = aggregated_data['scenes'][0][0] if aggregated_data['scenes'] else "room"
        motion_level = aggregated_data['avg_motion']
        
        if motion_level > 0.3:
            if "street" in primary_scene or "road" in primary_scene:
                base_prompts.append(types.WeightedPrompt(text="Electronic", weight=0.9))
            elif "park" in primary_scene or "outdoor" in primary_scene:
                base_prompts.append(types.WeightedPrompt(text="Indie Folk", weight=0.8))
            else:
                base_prompts.append(types.WeightedPrompt(text="Contemporary R&B", weight=0.8))
        else:
            if "room" in primary_scene or "indoor" in primary_scene:
                base_prompts.append(types.WeightedPrompt(text="Lo-Fi Hip Hop", weight=0.9))
            else:
                base_prompts.append(types.WeightedPrompt(text="Ambient", weight=0.8))
        
        # Instrument based on objects and environment
        key_objects = [obj[0] for obj in aggregated_data['objects'][:3]]
        if any("person" in obj for obj in key_objects):
            base_prompts.append(types.WeightedPrompt(text="Warm Acoustic Guitar", weight=0.8))
        if any("car" in obj or "vehicle" in obj for obj in key_objects):
            base_prompts.append(types.WeightedPrompt(text="Synth Pads", weight=0.7))
        
        # Mood based on lighting and depth
        if aggregated_data['brightness'] == "bright":
            base_prompts.append(types.WeightedPrompt(text="Upbeat", weight=0.8))
        elif aggregated_data['brightness'] == "dark":
            base_prompts.append(types.WeightedPrompt(text="Dreamy", weight=0.8))
        else:
            base_prompts.append(types.WeightedPrompt(text="Chill", weight=0.7))
        
        # Add depth-based prompt
        if aggregated_data['depth_profile'] in ['deep', 'deep_with_detail']:
            base_prompts.append(types.WeightedPrompt(text="Ethereal Ambience", weight=0.7))
        
        return base_prompts[:6]  # Limit to 6 base prompts

    def generate_dynamic_prompts(self, analysis: Dict) -> List[types.WeightedPrompt]:
        """
        Generate dynamic prompts that change based on current conditions
        These complement the stable base prompts
        """
        dynamic_prompts = []
        
        # Motion-based dynamic prompts
        current_motion = analysis.get('motion', 0)
        self.motion_window.append(current_motion)
        if len(self.motion_window) > 10:
            self.motion_window.pop(0)
        
        smoothed_motion = np.mean(self.motion_window)
        
        # Motion intensity prompts
        if smoothed_motion > 0.4:
            dynamic_prompts.append(types.WeightedPrompt(text="Fat Beats", weight=0.6))
        elif smoothed_motion > 0.2:
            dynamic_prompts.append(types.WeightedPrompt(text="Tight Groove", weight=0.5))
        else:
            dynamic_prompts.append(types.WeightedPrompt(text="Sustained Chords", weight=0.4))
        
        # Brightness change prompts
        current_brightness = analysis['brightness']
        self.brightness_history.append(current_brightness)
        if len(self.brightness_history) > 5:
            self.brightness_history.pop(0)
        
        # Add lighting-based dynamic prompt
        if len(set(self.brightness_history)) > 1:  # Brightness is changing
            if current_brightness == "bright":
                dynamic_prompts.append(types.WeightedPrompt(text="Bright Tones", weight=0.5))
            elif current_brightness == "dark":
                dynamic_prompts.append(types.WeightedPrompt(text="Subdued Melody", weight=0.5))
        
        # Object-based dynamic prompts
        current_objects = [obj['name'] for obj in analysis.get('objects', [])]
        if 'person' in current_objects:
            dynamic_prompts.append(types.WeightedPrompt(text="Live Performance", weight=0.4))
        
        # Depth-based dynamic prompts
        depth_info = analysis.get('depth', {})
        if depth_info:
            depth_profile = depth_info.get('depth_profile', 'medium')
            if depth_profile in ['deep', 'deep_with_detail']:
                dynamic_prompts.append(types.WeightedPrompt(text="Echo", weight=0.5))
            elif depth_profile == 'shallow':
                dynamic_prompts.append(types.WeightedPrompt(text="Crunchy Distortion", weight=0.4))
        
        return dynamic_prompts

    async def generate_prompts(self, analysis: Dict, is_video: bool) -> Tuple[List[types.WeightedPrompt], Dict]:
        """
        Main method to generate prompts - handles both base prompt generation and dynamic updates
        """
        config_updates = {}
        
        # Phase 1: Collect initial data and generate base prompts (first 5 seconds)
        if not self.base_prompts_generated:
            collection_complete = self.collect_initial_analysis(analysis)
            
            if collection_complete:
                print("5-second analysis complete. Generating base prompts with AI...")
                aggregated_data = self.aggregate_analysis_data()
                self.base_prompts = await self.generate_base_prompts_with_ai(aggregated_data)
                self.base_prompts_generated = True
                print(f"Base prompts established: {len(self.base_prompts)} prompts")
                
                # Initial config based on aggregated data
                avg_motion = aggregated_data.get('avg_motion', 0.5)
                config_updates['density'] = 0.4 + (avg_motion * 0.4)  # 0.4-0.8 range
                
                if aggregated_data.get('brightness') == "dark":
                    config_updates['brightness'] = 0.4
                elif aggregated_data.get('brightness') == "bright":
                    config_updates['brightness'] = 0.8
                else:
                    config_updates['brightness'] = 0.6
            
            # During collection phase, return simple prompts
            if not self.base_prompts_generated:
                temp_prompts = [
                    types.WeightedPrompt(text="Ambient", weight=0.8),
                    types.WeightedPrompt(text="Chill", weight=0.6)
                ]
                return temp_prompts, config_updates
        
        # Phase 2: Live processing with base + dynamic prompts
        all_prompts = []
        
        # Add stable base prompts (these never change, only weights might adjust slightly)
        for base_prompt in self.base_prompts:
            all_prompts.append(base_prompt)
        
        # Add dynamic prompts that respond to current conditions
        dynamic_prompts = self.generate_dynamic_prompts(analysis)
        all_prompts.extend(dynamic_prompts)
        
        # Update configuration based on current conditions
        if is_video:
            current_motion = analysis.get('motion', 0)
            self.motion_window.append(current_motion)
            if len(self.motion_window) > 10:
                self.motion_window.pop(0)
            
            smoothed_motion = np.mean(self.motion_window)
            
            # Adjust density based on motion
            base_density = 0.5
            motion_adjustment = smoothed_motion * 0.3
            config_updates['density'] = base_density + motion_adjustment
            
            # Adjust brightness based on current lighting
            current_brightness = analysis.get('brightness', 'medium')
            if current_brightness == "dark":
                config_updates['brightness'] = 0.4
            elif current_brightness == "bright":
                config_updates['brightness'] = 0.8
            else:
                config_updates['brightness'] = 0.6
        
        self.current_dynamic_prompts = dynamic_prompts
        return all_prompts, config_updates

    def get_prompt_status(self) -> Dict:
        """
        Get current status of prompt generation
        """
        return {
            'base_prompts_generated': self.base_prompts_generated,
            'num_base_prompts': len(self.base_prompts),
            'num_dynamic_prompts': len(self.current_dynamic_prompts),
            'analysis_frames_collected': len(self.initial_analysis_data),
            'collection_phase_complete': self.base_prompts_generated
        }

async def live_camera_processing(broadcast_func: Callable[[bytes], Awaitable[None]]):
    api_key = os.environ.get("LYRIA_API_KEY") or input("Enter API Key: ").strip()
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
    media_analyzer = MediaAnalyzer()
    prompt_generator = EnhancedMusicGenerator()
    
    # Camera setup with timeout and retry
    cap = None
    for _ in range(3):  # Try 3 times to open camera
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
            await session.set_music_generation_config(config=config)
            print("Initial configuration set")
            
            # Start playback
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
                    
                    # Analysis and prompt generation
                    analysis = media_analyzer.analyze_media(frame, is_video=True)
                    current_time = time.time()
                    
                    # Get prompt status
                    status = prompt_generator.get_prompt_status()
                    
                    # Update prompts and config
                    if not status['base_prompts_generated']:
                        # During initial collection phase
                        new_prompts, config_updates = await prompt_generator.generate_prompts(
                            analysis, is_video=True)
                        
                        print(f"Collection phase: {status['analysis_frames_collected']}/~150 frames")
                        
                    else:
                        # During live processing phase
                        if current_time - last_update_time > 2.0:  # Update every 2 seconds
                            new_prompts, config_updates = await prompt_generator.generate_prompts(
                                analysis, is_video=True)
                            
                            # Update configuration
                            if config_updates:
                                for key, value in config_updates.items():
                                    setattr(current_config, key, value)
                                await session.set_music_generation_config(config=current_config)
                                print("Updated config:", config_updates)
                            
                            # Update prompts
                            print(f"\n=== CURRENT PROMPTS (Base: {status['num_base_prompts']}, Dynamic: {status['num_dynamic_prompts']}) ===")
                            for i, prompt in enumerate(new_prompts, 1):
                                prompt_type = "BASE" if i <= status['num_base_prompts'] else "DYNAMIC"
                                print(f"{i}. [{prompt_type}] {prompt.text} (weight: {prompt.weight:.2f})")
                            
                            await session.set_weighted_prompts(prompts=new_prompts)
                            last_update_time = current_time
                    
                    # Display analysis
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
    object_names = ', '.join([obj['name'] for obj in analysis['objects']]) if analysis['objects'] else 'None'
    depth_info = analysis.get('depth', {})
    
    info = [
        f"=== ANALYSIS ===",
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
        f"=== PROMPT STATUS ===",
        f"Phase: {'COLLECTION' if not status['base_prompts_generated'] else 'LIVE PROCESSING'}",
        f"Base Prompts: {status['num_base_prompts']}",
        f"Dynamic Prompts: {status['num_dynamic_prompts']}",
        f"Frames Collected: {status['analysis_frames_collected']}"
    ]
    
    y = 20
    for line in info:
        if line.startswith("==="):
            color = (0, 255, 255)  # Yellow for headers
        elif line.startswith("Phase:"):
            color = (0, 255, 0) if status['base_prompts_generated'] else (0, 165, 255)  # Green/Orange
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
    
    cv2.imshow("Enhanced Media Analyzer", frame)
    cv2.waitKey(1)

async def start_live_processing(broadcast_func: Callable[[bytes], Awaitable[None]]):
    """Wrapper for live camera processing"""
    try:
        await live_camera_processing(broadcast_func)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}