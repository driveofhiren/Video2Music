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
import math

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

# Motion threshold - CRITICAL FIX
MOTION_UPDATE_THRESHOLD = 0.10  # Don't update prompts if motion < 0.10

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
    triggers: List[str]
    intensity_range: Tuple[float, float]
    max_duration: Optional[float] = None
    cooldown: float = 0.0
    priority: int = 1  # Higher number = higher priority

@dataclass
class ActivePrompt:
    prompt: DynamicPrompt
    current_weight: float
    activation_time: float
    target_weight: float
    fade_duration: float = 2.0
    decay_rate: float = 0.1  # How fast it decays when not triggered

class AdvancedTriggerEngine:
    """Enhanced trigger detection with intelligent analysis"""
    
    def __init__(self):
        self.motion_history = []
        self.brightness_history = []
        self.scene_history = []
        self.object_history = []
        self.trigger_cooldowns = {}
        self.energy_accumulator = 0.0
        self.beat_timer = 0.0
        self.last_major_change = 0.0
        
        # Musical timing
        self.bpm_estimate = 120
        self.beat_interval = 60.0 / self.bpm_estimate
        
    def update_histories(self, analysis: Dict):
        """Update all analysis histories"""
        current_time = time.time()
        
        # Motion history with timestamps
        motion = analysis.get('motion', 0.0)
        self.motion_history.append((current_time, motion))
        if len(self.motion_history) > 50:  # Keep 50 frames of history
            self.motion_history.pop(0)
            
        # Brightness history
        brightness_values = {'dark': 0.2, 'dim': 0.4, 'medium': 0.6, 'bright': 1.0}
        brightness_val = brightness_values.get(analysis.get('brightness', 'medium'), 0.6)
        self.brightness_history.append((current_time, brightness_val))
        if len(self.brightness_history) > 30:
            self.brightness_history.pop(0)
            
        # Scene history
        top_scene = analysis['top_scenes'][0][0] if analysis['top_scenes'] else 'unknown'
        self.scene_history.append((current_time, top_scene))
        if len(self.scene_history) > 20:
            self.scene_history.pop(0)
            
        # Object history
        object_names = [obj['name'] for obj in analysis.get('objects', [])]
        self.object_history.append((current_time, set(object_names)))
        if len(self.object_history) > 15:
            self.object_history.pop(0)
    
    def calculate_motion_derivatives(self) -> Dict[str, float]:
        """Calculate motion velocity and acceleration"""
        if len(self.motion_history) < 3:
            return {'velocity': 0.0, 'acceleration': 0.0, 'smoothed_motion': 0.0}
            
        # Get recent motion values
        recent_motions = [m[1] for m in self.motion_history[-10:]]
        smoothed_motion = np.mean(recent_motions)
        
        # Calculate velocity (rate of change)
        if len(self.motion_history) >= 5:
            recent_5 = [m[1] for m in self.motion_history[-5:]]
            velocity = np.gradient(recent_5)[-1] if len(recent_5) > 1 else 0.0
        else:
            velocity = 0.0
            
        # Calculate acceleration (rate of velocity change)
        if len(self.motion_history) >= 8:
            recent_8 = [m[1] for m in self.motion_history[-8:]]
            velocities = np.gradient(recent_8)
            acceleration = np.gradient(velocities)[-1] if len(velocities) > 1 else 0.0
        else:
            acceleration = 0.0
            
        return {
            'velocity': float(velocity),
            'acceleration': float(acceleration),
            'smoothed_motion': float(smoothed_motion)
        }
    
    def detect_scene_transitions(self) -> Dict[str, bool]:
        """Detect various types of scene transitions"""
        if len(self.scene_history) < 3:
            return {'scene_change': False, 'gradual_change': False, 'dramatic_shift': False}
            
        current_time = time.time()
        recent_scenes = [s[1] for s in self.scene_history[-5:]]
        
        # Sudden scene change
        scene_change = len(set(recent_scenes[-2:])) > 1
        
        # Gradual change over time
        gradual_change = len(set(recent_scenes)) > 2
        
        # Dramatic shift (completely different scene)
        if len(self.scene_history) >= 8:
            older_scenes = set([s[1] for s in self.scene_history[-8:-4]])
            newer_scenes = set([s[1] for s in self.scene_history[-4:]])
            dramatic_shift = len(older_scenes.intersection(newer_scenes)) == 0
        else:
            dramatic_shift = False
            
        return {
            'scene_change': scene_change,
            'gradual_change': gradual_change,
            'dramatic_shift': dramatic_shift
        }
    
    def calculate_energy_level(self, analysis: Dict) -> Dict[str, float]:
        """Calculate various energy metrics"""
        motion_data = self.calculate_motion_derivatives()
        current_motion = motion_data['smoothed_motion']
        
        # Energy accumulator (builds up with sustained motion)
        if current_motion > 0.25:
            self.energy_accumulator = min(1.0, self.energy_accumulator + 0.1)
        else:
            self.energy_accumulator = max(0.0, self.energy_accumulator - 0.05)
            
        # Motion spike detection
        motion_spike = (motion_data['velocity'] > 0.15 and 
                       current_motion > 0.4 and 
                       motion_data['acceleration'] > 0.1)
        
        # Calculate object-based energy
        objects = analysis.get('objects', [])
        action_objects = ['person', 'car', 'truck', 'motorcycle', 'bicycle', 'sports ball']
        object_energy = sum(1 for obj in objects if obj['name'] in action_objects) / max(len(objects), 1)
        
        return {
            'base_energy': current_motion,
            'accumulated_energy': self.energy_accumulator,
            'motion_spike': 1.0 if motion_spike else 0.0,
            'object_energy': object_energy,
            'combined_energy': (current_motion * 0.4 + self.energy_accumulator * 0.3 + object_energy * 0.3)
        }
    
    def detect_musical_moments(self) -> Dict[str, bool]:
        """Detect musically relevant moments"""
        current_time = time.time()
        
        # Beat sync (more intelligent timing)
        time_since_beat = current_time - self.beat_timer
        beat_sync = time_since_beat >= self.beat_interval * 0.9  # Allow slight timing flexibility
        
        if beat_sync:
            self.beat_timer = current_time
            
        # Beat emphasis (every 4 beats)
        beat_emphasis = beat_sync and (int(current_time / self.beat_interval) % 4 == 0)
        
        return {
            'beat_sync': beat_sync,
            'beat_emphasis': beat_emphasis,
            'drum_moment': beat_emphasis or (beat_sync and self.energy_accumulator > 0.6)
        }
    
    def check_trigger_cooldown(self, trigger_name: str, cooldown_time: float = 2.0) -> bool:
        """Check if trigger is in cooldown"""
        current_time = time.time()
        last_triggered = self.trigger_cooldowns.get(trigger_name, 0)
        
        if current_time - last_triggered >= cooldown_time:
            self.trigger_cooldowns[trigger_name] = current_time
            return True
        return False
    
    def generate_intelligent_triggers(self, analysis: Dict) -> Set[str]:
        """Generate comprehensive and intelligent triggers - SIMPLIFIED VERSION"""
        triggers = set()
        current_time = time.time()
        
        # Update all histories first
        self.update_histories(analysis)
        
        # CRITICAL: Don't update triggers if motion is too low
        current_motion = analysis.get('motion', 0.0)
        if current_motion < MOTION_UPDATE_THRESHOLD:
            # Return minimal triggers for low motion
            triggers.add('calm_scene')
            return triggers
        
        # Get calculated metrics
        motion_data = self.calculate_motion_derivatives()
        scene_transitions = self.detect_scene_transitions()
        energy_data = self.calculate_energy_level(analysis)
        musical_moments = self.detect_musical_moments()
        
        # === SIMPLIFIED MOTION-BASED TRIGGERS (ONLY 3) ===
        smoothed_motion = motion_data['smoothed_motion']
        
        if smoothed_motion > 0.25:
            triggers.add('high_motion')
        elif smoothed_motion > 0.15:
            triggers.add('medium_motion')
        else:
            triggers.add('calm_scene')
            
        # === SCENE-BASED TRIGGERS ===
        if scene_transitions['scene_change'] and self.check_trigger_cooldown('scene_change', 3.0):
            triggers.add('scene_change')
        if scene_transitions['dramatic_shift'] and self.check_trigger_cooldown('dramatic_shift', 5.0):
            triggers.add('dramatic_shift')
            
        # === ENERGY-BASED TRIGGERS ===
        if energy_data['accumulated_energy'] > 0.7:
            triggers.add('high_energy')
        if energy_data['motion_spike'] > 0:
            triggers.add('energy_spike')
            
        # === MUSICAL TIMING TRIGGERS ===
        if musical_moments['beat_sync']:
            triggers.add('beat_sync')
        if musical_moments['beat_emphasis']:
            triggers.add('beat_emphasis')
        if musical_moments['drum_moment']:
            triggers.add('drum_moment')
            
        # === ENVIRONMENTAL TRIGGERS ===
        brightness = analysis.get('brightness', 'medium')
        if brightness in ['dark', 'dim']:
            triggers.add('dark_scene')
        elif brightness == 'bright':
            triggers.add('bright_scene')
            
        # === DEPTH-BASED TRIGGERS ===
        depth_info = analysis.get('depth', {})
        depth_profile = depth_info.get('depth_profile', 'medium')
        avg_depth = depth_info.get('avg_depth', 0.5)
        
        if 'deep' in depth_profile or avg_depth > 0.7:
            triggers.add('deep_scene')
        elif 'shallow' in depth_profile or avg_depth < 0.3:
            triggers.add('close_scene')
            
        # === OBJECT-BASED TRIGGERS ===
        objects = analysis.get('objects', [])
        object_names = [obj['name'] for obj in objects]
        
        # Human presence
        if any(name == 'person' for name in object_names):
            triggers.add('human_presence')
            
        # Vehicles and action
        vehicle_objects = ['car', 'truck', 'motorcycle', 'bicycle']
        if any(name in vehicle_objects for name in object_names):
            triggers.add('vehicle_scene')
            
        # Sports and dynamic objects
        dynamic_objects = ['sports ball', 'frisbee', 'skateboard']
        if any(name in dynamic_objects for name in object_names):
            triggers.add('sports_scene')
            
        return triggers

class MediaAnalyzer:
    def __init__(self):
        self.places_model = self._load_places_model().to(device)
        self.classes = self._load_categories()
        self.prev_gray = None
        # Initialize depth estimation model
        self.depth_model = self._load_depth_model().to(device)
        self.depth_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def _load_depth_model(self):
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
        depth = self.analyze_depth(frame_resized)
        
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
        depth = self.analyze_depth(img_resized)
        
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
        self.dynamic_prompt_pools = {}
        self.base_prompts_generated = False
        
        # Enhanced trigger engine
        self.trigger_engine = AdvancedTriggerEngine()
        
        # Live processing state
        self.active_prompts = {}
        self.prompt_history = {}
        self.scene_context_history = []
        
        # Analysis aggregation for first 5 seconds
        self.initial_analysis_data = []
        self.analysis_start_time = None
        
        # Musical balance constraints - Enhanced
        self.max_concurrent_prompts = {
            PromptCategory.RHYTHMIC: 3,
            PromptCategory.BASS: 2,
            PromptCategory.MELODIC: 2,
            PromptCategory.PERCUSSIVE: 4,
            PromptCategory.ATMOSPHERIC: 2,
            PromptCategory.TRANSITION: 1,
            PromptCategory.TEXTURAL: 2,
            PromptCategory.HARMONIC: 2,
            PromptCategory.DRUMS: 3
        }
        
        # Prompt priority weights by category
        self.category_priorities = {
            PromptCategory.DRUMS: 1.3,       # Higher priority for drums
            PromptCategory.BASS: 1.2,        # Bass is important
            PromptCategory.RHYTHMIC: 1.1,    # Rhythm is key
            PromptCategory.MELODIC: 1.0,     # Standard priority
            PromptCategory.ATMOSPHERIC: 0.9,  # Slightly lower
            PromptCategory.HARMONIC: 0.9,
            PromptCategory.PERCUSSIVE: 0.8,
            PromptCategory.TEXTURAL: 0.7,
            PromptCategory.TRANSITION: 1.5   # Highest when active
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
You are an electronic dance music AI DJ. Based on this comprehensive video scene analysis, generate a complete adaptive music system including:
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


Available Enhanced Triggers:
Motion: high_motion, medium_motion, low_motion, calm_moment, motion_acceleration, motion_spike, energy_burst, peak_energy
Scene: scene_change, dramatic_shift, gradual_transition, deep_scene, shallow_scene, spatial_depth, intimate_space
Energy: high_energy_sustained, energy_spike, peak_energy, action_scene, sports_scene, dynamic_action
Musical: beat_sync, beat_emphasis, drum_moment, harmonic_need, harmonic_support, texture_need
Environmental: dark_scene, bright_scene, atmospheric_mood, energetic_mood, evening_mood, morning_energy
Spatial: deep_scene, shallow_scene, spatial_depth, close_focus, atmospheric_space
Social: human_presence, social_scene, vehicle_scene, movement_scene


STRICT scale_name SELECTION RULES for config > scale:
- MUST select exactly ONE scale from this EXACT list (copy/paste the name exactly as shown):
  "C_MAJOR_A_MINOR", "D_FLAT_MAJOR_B_FLAT_MINOR", "D_MAJOR_B_MINOR", 
  "E_FLAT_MAJOR_C_MINOR", "E_MAJOR_D_FLAT_MINOR", "F_MAJOR_D_MINOR", 
  "G_FLAT_MAJOR_E_FLAT_MINOR", "G_MAJOR_E_MINOR", "A_FLAT_MAJOR_F_MINOR", 
  "A_MAJOR_G_FLAT_MINOR", "B_FLAT_MAJOR_G_MINOR", "B_MAJOR_A_FLAT_MINOR"

- DO NOT invent new scale names
- DO NOT use abbreviations
- DO NOT modify the formatting
- do not give scale name like A_MINOR_C_MAJOR

Generate JSON in this exact format:
{{
    "config": {{
        "bpm": <60-200>,
        "scale": "<scale_name>",
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
                "triggers": ["high_motion", "beat_sync", "energy_spike"],
                "intensity_range": [0.5, 2.0],
                "priority": <1-3>
            }}
        ],
        "bass": [
            {{
                "text": "<bass element like sub bass, bass hit, bass drop>",
                "base_weight": <1.0-4.0>,
                "triggers": ["motion_spike", "scene_change", "peak_energy"],
                "intensity_range": [0.8, 2.5],
                "priority": <1-3>
            }}
        ],
        "melodic": [
            {{
                "text": "<melodic element like lead synth, pad, arpeggio>",
                "base_weight": <1.0-3.5>,
                "triggers": ["calm_moment", "harmonic_need", "atmospheric_mood"],
                "intensity_range": [0.5, 2.0],
                "priority": <1-3>
            }}
        ],
        "percussive": [
            {{
                "text": "<percussion like shaker, clap, rim shot>",
                "base_weight": <0.8-3.0>,
                "triggers": ["medium_motion", "beat_emphasis", "texture_need"],
                "intensity_range": [0.3, 1.8],
                "priority": <1-2>
            }}
        ],
        "atmospheric": [
            {{
                "text": "<atmosphere like reverb, ambient texture, pad>",
                "base_weight": <1.0-3.0>,
                "triggers": ["deep_scene", "calm_moment", "spatial_depth"],
                "intensity_range": [0.5, 1.5],
                "priority": <1-2>
            }}
        ],
        "transition": [
            {{
                "text": "<transition like build-up, drop, sweep>",
                "base_weight": <2.0-4.0>,
                "triggers": ["scene_change", "dramatic_shift", "peak_energy"],
                "intensity_range": [1.0, 3.0],
                "max_duration": 4.0,
                "cooldown": 8.0,
                "priority": 3
            }}
        ],
        "textural": [
            {{
                "text": "<texture like vinyl crackle, tape saturation, distortion>",
                "base_weight": <0.5-2.0>,
                "triggers": ["texture_need", "atmospheric_mood", "evening_mood"],
                "intensity_range": [0.2, 1.2],
                "priority": <1-2>
            }}
        ],
        "harmonic": [
            {{
                "text": "<harmony like chord stab, sustained note, harmonic layer>",
                "base_weight": <1.0-3.0>,
                "triggers": ["harmonic_support", "calm_moment", "spatial_depth"],
                "intensity_range": [0.5, 2.0],
                "priority": <1-2>
            }}
        ],
        "drums": [
            {{
                "text": "<drum kit element like kick drum, snare hit, cymbal crash, hi-hat pattern>",
                "base_weight": <1.5-4.0>,
                "triggers": ["high_motion", "action_scene", "energy_spike", "drum_moment", "beat_emphasis"],
                "intensity_range": [0.8, 3.5],
                "priority": <2-3>
            }}
        ]
    }}
}}

Guidelines:
- Choose BPM based on motion level and scene energy (high motion = higher BPM)
- Base prompts (4-6): Core elements that define the musical style and stay constant, mood weight should be 1.0-2.0
- Dynamic pools: Each category should have 4-8 contextually relevant options
- DRUMS category: Focus on expressive drum kit elements that respond to video action and energy
- Higher priority prompts (2-3) are selected first when multiple triggers match
- Transition elements need duration limits and cooldowns to prevent chaos
- Use the enhanced trigger list for more intelligent responses
- Consider depth and spatial elements for atmospheric choices
- Match energy level of prompts to scene characteristics
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
            
            # Process dynamic prompt pools with enhanced features
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
                            cooldown=prompt_data.get('cooldown', 0.0),
                            priority=prompt_data.get('priority', 1)
                        )
                        self.dynamic_prompt_pools[category].append(dynamic_prompt)
                        print(f"Added {category_name} prompt: {dynamic_prompt.text} (priority: {dynamic_prompt.priority})")
                
                except ValueError as e:
                    print(f"Unknown category {category_name}, skipping: {e}")
                    continue
            
            total_dynamic_prompts = sum(len(prompts) for prompts in self.dynamic_prompt_pools.values())
            print(f"Successfully generated: {len(base_prompts)} base prompts, {total_dynamic_prompts} dynamic prompts across {len(self.dynamic_prompt_pools)} categories")
            
            return config, base_prompts
                
        except Exception as e:
            print(f"Error generating config and prompts: {e}")
            return self._fallback_base_config(aggregated_data), self._fallback_base_prompts(aggregated_data)

    def _calculate_dynamic_intensity(self, analysis: Dict, prompt: DynamicPrompt, triggers: Set[str]) -> float:
        """Enhanced intensity calculation with more sophisticated logic"""
        intensity = 1.0
        
        # Base intensity from trigger matching
        matching_triggers = set(prompt.triggers).intersection(triggers)
        if not matching_triggers:
            return 0.0  # No triggers matched
            
        trigger_strength = len(matching_triggers) / len(prompt.triggers)
        
        # Motion influence with smoothing
        motion = analysis.get('motion', 0.0)
        
        # Category-specific motion responses
        if prompt.category in [PromptCategory.RHYTHMIC, PromptCategory.BASS, PromptCategory.DRUMS]:
            intensity *= (0.4 + motion * 1.6)  # Strong response to motion
        elif prompt.category == PromptCategory.ATMOSPHERIC:
            intensity *= (1.3 - motion * 0.5)  # Inverse response to motion
        elif prompt.category == PromptCategory.MELODIC:
            intensity *= (0.7 + motion * 0.6)  # Moderate response
        
        # Special intensity boosts based on specific triggers
        if prompt.category == PromptCategory.DRUMS:
            if 'high_motion' in triggers or 'action_scene' in triggers:
                intensity *= 1.4
            if 'energy_spike' in triggers or 'peak_energy' in triggers:
                intensity *= 1.3
            if 'drum_moment' in triggers:
                intensity *= 1.2
                
        if prompt.category == PromptCategory.TRANSITION:
            if 'dramatic_shift' in triggers:
                intensity *= 1.5
            if 'scene_change' in triggers:
                intensity *= 1.3
        
        # Depth influence
        depth_info = analysis.get('depth', {})
        avg_depth = depth_info.get('avg_depth', 0.5)
        
        if prompt.category == PromptCategory.ATMOSPHERIC:
            intensity *= (0.6 + avg_depth * 1.0)  # Deeper scenes boost atmosphere
        elif prompt.category == PromptCategory.DRUMS and avg_depth < 0.3:
            intensity *= 1.15  # Close-up scenes boost drums
            
        # Priority weighting
        priority_multiplier = 1.0 + (prompt.priority - 1) * 0.2
        intensity *= priority_multiplier
        
        # Apply trigger strength
        intensity *= (0.2 + trigger_strength * 1.3)
        
        # Clamp to prompt's intensity range
        min_intensity, max_intensity = prompt.intensity_range
        intensity = max(min_intensity, min(max_intensity, intensity))
        
        return intensity

    def _select_dynamic_prompts(self, analysis: Dict) -> List[types.WeightedPrompt]:
        """Enhanced dynamic prompt selection with intelligent prioritization"""
        if not self.dynamic_prompt_pools:
            return []
        
        current_time = time.time()
        
        # CRITICAL: Check motion threshold before processing
        current_motion = analysis.get('motion', 0.0)
        if current_motion < MOTION_UPDATE_THRESHOLD:
            print(f"Motion too low ({current_motion:.3f} < {MOTION_UPDATE_THRESHOLD}), skipping prompt update")
            # Return previous prompts or minimal set
            if hasattr(self, '_last_low_motion_prompts'):
                return self._last_low_motion_prompts
            else:
                # Minimal prompts for low motion
                minimal_prompts = [
                    types.WeightedPrompt(text="ambient pad", weight=1.2),
                    types.WeightedPrompt(text="soft reverb", weight=0.8)
                ]
                self._last_low_motion_prompts = minimal_prompts
                return minimal_prompts
        
        # Get intelligent triggers
        triggers = self.trigger_engine.generate_intelligent_triggers(analysis)
        selected_prompts = []
        
        print(f"Active triggers: {sorted(list(triggers))}")
        
        # Collect all candidates with their scores
        all_candidates = []
        
        for category, prompts in self.dynamic_prompt_pools.items():
            for prompt in prompts:
                # Check cooldown
                last_activation = self.prompt_history.get(prompt.text, 0)
                if current_time - last_activation < prompt.cooldown:
                    continue
                
                # Check if triggers match
                matching_triggers = set(prompt.triggers).intersection(triggers)
                if not matching_triggers:
                    continue
                
                # Calculate comprehensive score
                trigger_score = len(matching_triggers) / len(prompt.triggers)
                intensity = self._calculate_dynamic_intensity(analysis, prompt, triggers)
                
                if intensity > 0:
                    # Enhanced scoring with category priority
                    category_weight = self.category_priorities.get(category, 1.0)
                    priority_bonus = prompt.priority * 0.3
                    final_score = trigger_score * intensity * category_weight + priority_bonus
                    
                    all_candidates.append({
                        'prompt': prompt,
                        'category': category,
                        'score': final_score,
                        'intensity': intensity,
                        'trigger_score': trigger_score
                    })
        
        # Sort by score and select best ones per category
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        category_selections = {cat: [] for cat in PromptCategory}
        
        for candidate in all_candidates:
            category = candidate['category']
            max_concurrent = self.max_concurrent_prompts.get(category, 2)
            
            if len(category_selections[category]) < max_concurrent:
                if candidate['score'] > 0.4:  # Minimum quality threshold
                    category_selections[category].append(candidate)
        
        # Convert to final prompts
        for category, candidates in category_selections.items():
            for candidate in candidates:
                prompt = candidate['prompt']
                intensity = candidate['intensity']
                final_weight = prompt.base_weight * intensity
                
                selected_prompts.append(types.WeightedPrompt(
                    text=prompt.text,
                    weight=final_weight
                ))
                
                # Update prompt history
                self.prompt_history[prompt.text] = current_time
                
                print(f"Selected {category.value}: {prompt.text} "
                      f"(weight: {final_weight:.2f}, score: {candidate['score']:.2f}, "
                      f"priority: {prompt.priority})")
        
        # Store current dynamic prompts for display
        self.current_dynamic_prompts = selected_prompts
        
        return selected_prompts

    async def generate_prompts(self, analysis: Dict, is_video: bool) -> Tuple[List[types.WeightedPrompt], Dict]:
        """Main method to generate prompts with enhanced logic"""
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
        
        # Enhanced configuration updates based on analysis
        if is_video:
            # Motion-based updates with smoothing
            motion = analysis.get('motion', 0)
            if motion >= MOTION_UPDATE_THRESHOLD:  # Only update if motion is significant
                # Enhanced motion response
                motion_energy = self.trigger_engine.calculate_energy_level(analysis)
                base_density = 0.45
                
                # More sophisticated density calculation
                energy_factor = motion_energy.get('combined_energy', motion)
                density_adjustment = energy_factor * 0.35
                config_updates['density'] = max(0.15, min(0.85, base_density + density_adjustment))
                
                # BPM adjustments based on energy
                if hasattr(self, 'base_bpm'):
                    bpm_adjustment = int(energy_factor * 25)
                    config_updates['bpm'] = max(80, min(180, self.base_bpm + bpm_adjustment))
            
            # Brightness-based adjustments
            brightness = analysis.get('brightness', 'medium')
            brightness_map = {
                'dark': 1.2,
                'dim': 1.4,
                'medium': 1.6,
                'bright': 1.9
            }
            config_updates['brightness'] = brightness_map.get(brightness, 1.6)
            
            # Depth-based adjustments
            depth_info = analysis.get('depth', {})
            avg_depth = depth_info.get('avg_depth', 0.5)
            if avg_depth > 0.7:  # Deep scenes
                config_updates['guidance'] = 4.5  # More atmospheric guidance
            elif avg_depth < 0.3:  # Shallow/close scenes
                config_updates['guidance'] = 5.5  # More focused guidance
        
        return all_prompts, config_updates

    def get_prompt_status(self) -> Dict:
        """Get current status of the enhanced prompt generation system"""
        active_categories = {cat.value: len(prompts) for cat, prompts in self.dynamic_prompt_pools.items()}
        
        # Enhanced status information
        trigger_stats = {
            'motion_history_length': len(self.trigger_engine.motion_history),
            'energy_accumulator': self.trigger_engine.energy_accumulator,
            'active_cooldowns': len(self.trigger_engine.trigger_cooldowns)
        }
        
        return {
            'base_prompts_generated': self.base_prompts_generated,
            'num_base_prompts': len(self.base_prompts),
            'dynamic_pool_sizes': active_categories,
            'total_dynamic_prompts': sum(len(prompts) for prompts in self.dynamic_prompt_pools.values()),
            'num_current_dynamic': len(self.current_dynamic_prompts),
            'analysis_frames_collected': len(self.initial_analysis_data),
            'system_ready': self.base_prompts_generated and bool(self.dynamic_prompt_pools),
            'trigger_engine_stats': trigger_stats,
            'motion_threshold': MOTION_UPDATE_THRESHOLD
        }

    def _fallback_base_config(self, aggregated_data: Dict) -> types.LiveMusicGenerationConfig:
        """Enhanced fallback configuration"""
        avg_motion = aggregated_data.get('avg_motion', 0.5)
        
        config = types.LiveMusicGenerationConfig(
            bpm=int(100 + avg_motion * 60),
            scale=types.Scale.C_MAJOR_A_MINOR,
            density=0.4 + avg_motion * 0.3,
            brightness=1.6,
            guidance=5.0
        )
        
        # Store base BPM for later adjustments
        self.base_bpm = config.bpm
        return config

    def _fallback_base_prompts(self, aggregated_data: Dict) -> List[types.WeightedPrompt]:
        """Enhanced fallback base prompts with improved dynamic pools"""
        prompts = [
            types.WeightedPrompt(text="Electronic", weight=2.2),
            types.WeightedPrompt(text="Ambient", weight=1.8),
            types.WeightedPrompt(text="Chill", weight=1.5)
        ]
        
        # Enhanced fallback dynamic pools
        self.dynamic_prompt_pools = {
            PromptCategory.RHYTHMIC: [
                DynamicPrompt("kick drum", PromptCategory.RHYTHMIC, 2.5, 
                            ["high_motion", "beat_sync", "energy_spike"], (1.0, 3.0), priority=2),
                DynamicPrompt("snare pattern", PromptCategory.RHYTHMIC, 2.0, 
                            ["medium_motion", "beat_emphasis"], (0.8, 2.5), priority=2),
                DynamicPrompt("hi-hat groove", PromptCategory.RHYTHMIC, 1.8, 
                            ["beat_sync", "texture_need"], (0.6, 2.2), priority=1)
            ],
            PromptCategory.BASS: [
                DynamicPrompt("sub bass", PromptCategory.BASS, 2.8, 
                            ["motion_spike", "peak_energy", "action_scene"], (1.2, 3.2), priority=3),
                DynamicPrompt("bass hit", PromptCategory.BASS, 2.2, 
                            ["scene_change", "energy_burst"], (0.9, 2.8), priority=2)
            ],
            PromptCategory.ATMOSPHERIC: [
                DynamicPrompt("reverb wash", PromptCategory.ATMOSPHERIC, 1.8, 
                            ["deep_scene", "calm_moment", "spatial_depth"], (0.8, 2.0), priority=1),
                DynamicPrompt("ambient texture", PromptCategory.ATMOSPHERIC, 1.5, 
                            ["atmospheric_mood", "evening_mood"], (0.6, 1.8), priority=1)
            ],
            PromptCategory.DRUMS: [
                DynamicPrompt("kick drum punch", PromptCategory.DRUMS, 3.0, 
                            ["high_motion", "action_scene", "beat_emphasis"], (1.0, 3.8), priority=3),
                DynamicPrompt("snare crack", PromptCategory.DRUMS, 2.5, 
                            ["energy_spike", "drum_moment"], (0.8, 3.2), priority=2),
                DynamicPrompt("cymbal splash", PromptCategory.DRUMS, 2.8, 
                            ["scene_change", "dramatic_shift"], (1.2, 3.5), 
                            max_duration=3.0, cooldown=6.0, priority=3)
            ],
            PromptCategory.TRANSITION: [
                DynamicPrompt("sweep build", PromptCategory.TRANSITION, 3.5, 
                            ["dramatic_shift", "peak_energy"], (2.0, 4.0), 
                            max_duration=4.0, cooldown=10.0, priority=3)
            ]
        }
        
        return prompts

# Rest of the code remains the same for live_camera_processing and other functions...

async def live_camera_processing(broadcast_func: Callable[[bytes], Awaitable[None]]):
    """Process live camera feed and generate adaptive music - Enhanced version"""
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
                    
                    # Apply initial config updates and store base BPM
                    if config_updates:
                        for attr, value in config_updates.items():
                            setattr(config, attr, value)
                        prompt_generator.base_bpm = config.bpm  # Store for later adjustments
                        await session.set_music_generation_config(config=config)
                    
                    print(f"Base prompts set: {len(base_prompts)} prompts")
                    break
                
                await asyncio.sleep(0.1)

            # Start playback after base prompts are set
            await session.play()
            print("Playback started with enhanced trigger system")
            
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
                    
                    # Enhanced update logic with motion threshold check
                    if current_time - last_update_time > 3.0:  # Reduced update frequency
                        # Generate combined prompts (base + dynamic)
                        all_prompts, config_updates = await prompt_generator.generate_prompts(
                            analysis, is_video=True)
                        
                        # Only update if there are changes and motion is sufficient
                        should_update = (config_updates or 
                                       len(prompt_generator.current_dynamic_prompts) > 0)
                        
                        if should_update:
                            # Update configuration if needed
                            config_changed = False
                            if config_updates:
                                for key, value in config_updates.items():
                                    if hasattr(current_config, key):
                                        old_value = getattr(current_config, key)
                                        if abs(old_value - value) > 0.05 if isinstance(value, float) else old_value != value:
                                            setattr(current_config, key, value)
                                            config_changed = True
                                
                                if config_changed:
                                    await session.set_music_generation_config(config=current_config)
                                    print("Updated config:", {k: v for k, v in config_updates.items()})
                            
                            # Enhanced prompt display
                            print("\n" + "="*50)
                            print("🎵 ENHANCED MUSIC SYSTEM STATUS 🎵")
                            print("="*50)
                            
                            # Show motion threshold status
                            current_motion = analysis.get('motion', 0.0)
                            motion_status = "✅ ACTIVE" if current_motion >= MOTION_UPDATE_THRESHOLD else "⏸️  PAUSED"
                            print(f"Motion: {current_motion:.3f} ({motion_status})")
                            
                            # Show trigger engine stats
                            status = prompt_generator.get_prompt_status()
                            trigger_stats = status.get('trigger_engine_stats', {})
                            print(f"Energy Level: {trigger_stats.get('energy_accumulator', 0):.2f}")
                            print(f"Active Cooldowns: {trigger_stats.get('active_cooldowns', 0)}")
                            
                            print("\n📻 Base Prompts (Constant):")
                            for i, prompt in enumerate(base_prompts, 1):
                                print(f"  {i}. {prompt.text} (weight: {prompt.weight:.2f})")
                            
                            if prompt_generator.current_dynamic_prompts:
                                print("\n🎛️  Dynamic Prompts (Adaptive):")
                                for i, prompt in enumerate(prompt_generator.current_dynamic_prompts, 1):
                                    print(f"  {i}. {prompt.text} (weight: {prompt.weight:.2f})")
                            else:
                                print("\n🎛️  Dynamic Prompts: None active")
                            
                            print("="*50 + "\n")
                            
                            # Update prompts with combined base + dynamic
                            await session.set_weighted_prompts(prompts=all_prompts)
                            last_update_time = current_time
                    
                    # Enhanced display
                    status = prompt_generator.get_prompt_status()
                    display_enhanced_analysis(analysis, analysis['frame_resized'], status)
                    
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

def display_enhanced_analysis(analysis, frame, status):
    """Enhanced display function with improved trigger system information"""
    object_names = ', '.join([obj['name'] for obj in analysis['objects']]) if analysis['objects'] else 'None'
    depth_info = analysis.get('depth', {})
    trigger_stats = status.get('trigger_engine_stats', {})
    
    # Motion status with threshold check
    current_motion = analysis.get('motion', 0.0)
    motion_status = "ACTIVE" if current_motion >= MOTION_UPDATE_THRESHOLD else "PAUSED"
    motion_color = (0, 255, 0) if current_motion >= MOTION_UPDATE_THRESHOLD else (0, 165, 255)
    
    info = [
        f"=== ENHANCED SCENE ANALYSIS ===",
        f"Scene 1: {analysis['top_scenes'][0][0]} ({analysis['top_scenes'][0][1]*100:.1f}%)",
        f"Scene 2: {analysis['top_scenes'][1][0]} ({analysis['top_scenes'][1][1]*100:.1f}%)",
        f"Scene 3: {analysis['top_scenes'][2][0]} ({analysis['top_scenes'][2][1]*100:.1f}%)",
        f"Objects: {object_names}",
        f"Lighting: {analysis['brightness']}",
        f"Weather: {analysis['weather']}",
        f"Time: {analysis['time_of_day']}",
        f"Motion: {current_motion:.3f} ({motion_status})" if 'motion' in analysis else "Photo (no motion)",
        f"Motion Threshold: {MOTION_UPDATE_THRESHOLD}",
        f"Colors: {' '.join(analysis['colors'])}",
        f"Depth: {depth_info.get('depth_profile', 'N/A')}",
        f"",
        f"=== INTELLIGENT TRIGGER ENGINE ===",
        f"Phase: {'COLLECTION' if not status['base_prompts_generated'] else 'LIVE PROCESSING'}",
        f"Energy Level: {trigger_stats.get('energy_accumulator', 0):.2f}",
        f"Motion History: {trigger_stats.get('motion_history_length', 0)} frames",
        f"Active Cooldowns: {trigger_stats.get('active_cooldowns', 0)}",
        f"",
        f"=== MUSIC SYSTEM STATUS ===",
        f"Base Prompts: {status['num_base_prompts']}",
        f"Dynamic Pools: {sum(status['dynamic_pool_sizes'].values()) if 'dynamic_pool_sizes' in status else 0}",
        f"Active Dynamic: {status.get('num_current_dynamic', 0)}",
        f"System Ready: {'YES' if status.get('system_ready', False) else 'NO'}",
        f"",
        f"=== DYNAMIC CATEGORIES ===",
    ]
    
    # Add dynamic pool status with enhanced info
    if 'dynamic_pool_sizes' in status:
        for category, count in status['dynamic_pool_sizes'].items():
            info.append(f"{category.capitalize()}: {count} prompts")
    
    y = 20
    for line in info:
        if line.startswith("==="):
            color = (0, 255, 255)  # Yellow for headers
        elif line.startswith("Phase:"):
            color = (0, 255, 0) if status['base_prompts_generated'] else (0, 165, 255)
        elif line.startswith("System Ready:"):
            color = (0, 255, 0) if status.get('system_ready', False) else (0, 0, 255)
        elif line.startswith("Motion:") and 'motion' in analysis:
            color = motion_color
        elif line.startswith("Energy Level:"):
            energy = trigger_stats.get('energy_accumulator', 0)
            if energy > 0.7:
                color = (0, 100, 255)  # Red for high energy
            elif energy > 0.4:
                color = (0, 255, 255)  # Yellow for medium energy
            else:
                color = (255, 255, 255)  # White for low energy
        else:
            color = (255, 255, 255)  # White for regular info
            
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += 18
    
    # Enhanced depth map display
    if 'depth' in analysis and 'depth_map' in analysis['depth']:
        depth_map = analysis['depth']['depth_map']
        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        depth_map = cv2.resize(depth_map, (frame.shape[1] // 4, frame.shape[0] // 4))
        
        # Add border to depth map
        cv2.rectangle(depth_map, (0, 0), (depth_map.shape[1]-1, depth_map.shape[0]-1), (255, 255, 255), 2)
        
        # Overlay depth map on the frame
        frame[10:10+depth_map.shape[0], frame.shape[1]-depth_map.shape[1]-10:frame.shape[1]-10] = depth_map
    
    # Add motion visualization
    if current_motion > 0:
        # Draw motion bar
        bar_width = int(current_motion * 200)
        bar_color = motion_color
        cv2.rectangle(frame, (10, frame.shape[0] - 30), (10 + bar_width, frame.shape[0] - 10), bar_color, -1)
        cv2.rectangle(frame, (10, frame.shape[0] - 30), (210, frame.shape[0] - 10), (255, 255, 255), 2)
    
    cv2.imshow("Enhanced Adaptive Music System", frame)
    cv2.waitKey(1)

async def start_live_processing(broadcast_func: Callable[[bytes], Awaitable[None]]):
    """Wrapper for enhanced live camera processing"""
    try:
        await live_camera_processing(broadcast_func)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
