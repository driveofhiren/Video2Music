import torch
import torchvision.models as models
import torch.nn.functional as F
import cv2
import numpy as np
import os
import time
import asyncio
from datetime import datetime
from torchvision import transforms
from sklearn.cluster import KMeans
from ultralytics import YOLO

import pyaudio
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ---------- Video Analysis Functions ----------

def load_categories(filename='categories_places365.txt'):
    with open(filename) as f:
        return [line.strip().split(' ')[0][3:] for line in f if line.strip()]

def load_places_model():
    model = models.resnet18(num_classes=365)
    checkpoint = torch.load('resnet18_places365.pth.tar', map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return preprocess(img).unsqueeze(0)

def get_brightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    if brightness < 50:
        return "dark"
    elif brightness < 100:
        return "dim"
    elif brightness < 180:
        return "medium"
    return "bright"

def get_dominant_colors(frame, k=3):
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

def simulate_weather(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = np.mean(hsv[:, :, 0])
    brightness = np.mean(hsv[:, :, 2])
    saturation = np.mean(hsv[:, :, 1])
    
    if brightness < 60 and saturation > 40:
        return "rainy"
    elif hue > 90 and brightness > 160:
        return "sunny"
    elif 50 < hue < 90 and saturation < 60:
        return "cloudy"
    elif brightness > 200 and saturation < 30:
        return "fog"
    else:
        return "clear"

def get_time_of_day():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 20:
        return "evening"
    return "night"

def detect_motion(current_gray, prev_gray):
    if prev_gray is None:
        return 0
    diff = cv2.absdiff(current_gray, prev_gray)
    return np.mean(diff)

def detect_objects(yolo_model, frame):
    results = yolo_model(frame, verbose=False)[0]
    names = yolo_model.names
    detected_objects = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if float(conf) > 0.5:  # Only include confident detections
            detected_objects.append({
                'name': names[int(cls)],
                'confidence': float(conf)
            })
    return detected_objects

# ---------- Music Prompt Mapping Functions ----------

def map_to_music_prompt(top_scenes, objects, brightness, weather, time_of_day, motion, colors):
    # Initialize components
    instruments = []
    genres = []
    moods = []
    config_updates = {}
    
    # Scene analysis mapping
    scene_weights = [s[1] for s in top_scenes]
    primary_scene = top_scenes[0][0].lower()
    secondary_scene = top_scenes[1][0].lower() if scene_weights[1] > 0.2 else None
    
    # Scene mappings
    scene_mappings = {
        'beach': ('Steel Drum', 'Reggae', 'Chill'),
        'forest': ('Acoustic Guitar', 'Folk', 'Ethereal Ambience'),
        'city': ('Precision Bass', 'Synthpop', 'Upbeat'),
        'office': ('Rhodes Piano', 'Lo-Fi Hip Hop', 'Subdued Melody'),
        'mountain': ('Harp', 'Orchestral Score', 'Emotional'),
        'street': ('Trumpet', 'Jazz Fusion', 'Live Performance'),
        'kitchen': ('Marimba', 'Bossa Nova', 'Light'),
        'bedroom': ('Smooth Pianos', 'Ambient', 'Dreamy'),
        'restaurant': ('Vibraphone', 'Jazz', 'Sophisticated'),
        'park': ('Clarinet', 'Classical', 'Peaceful')
    }
    
    # Find best scene match
    matched_scene = None
    for scene_key in scene_mappings:
        if scene_key in primary_scene:
            matched_scene = scene_key
            break
    
    if matched_scene:
        instruments.append(scene_mappings[matched_scene][0])
        genres.append(scene_mappings[matched_scene][1])
        moods.append(scene_mappings[matched_scene][2])
    
    # Object mappings
    object_mappings = {
        'person': ('Vocal Samples', 0.3),
        'car': ('Dirty Synths', 0.4),
        'tree': ('Wind Chimes', 0.2),
        'dog': ('Whistles', 0.1),
        'cat': ('Purring Sounds', 0.1),
        'computer': ('Chiptune', 0.5),
        'phone': ('Electronic Beeps', 0.3),
        'book': ('Page Turning Sounds', 0.2),
        'chair': ('Wooden Percussion', 0.2),
        'cup': ('Glass Harmonica', 0.1)
    }
    
    for obj in objects:
        name = obj['name'].lower()
        for key in object_mappings:
            if key in name:
                instruments.append(object_mappings[key][0])
    
    # Brightness mapping
    brightness_mapping = {
        'dark': ('Spacey Synths', 'Ominous Drone', 0.3),
        'dim': ('Warm Acoustic Guitar', 'Ambient', 0.5),
        'medium': ('Rhodes Piano', 'Jazz Fusion', 0.7),
        'bright': ('Trumpet', 'Upbeat', 0.9)
    }
    
    if brightness in brightness_mapping:
        inst, mood, bright_val = brightness_mapping[brightness]
        instruments.append(inst)
        moods.append(mood)
        config_updates['brightness'] = bright_val
    
    # Weather mapping
    weather_mappings = {
        'sunny': ('Steel Drum', 'Upbeat'),
        'rainy': ('Piano', 'Melancholic'),
        'cloudy': ('Cello', 'Subdued Melody'),
        'fog': ('Synth Pads', 'Dreamy'),
        'clear': ('Acoustic Guitar', 'Peaceful')
    }
    
    if weather in weather_mappings:
        instruments.append(weather_mappings[weather][0])
        moods.append(weather_mappings[weather][1])
    
    # Time of day mapping
    time_mappings = {
        'morning': ('Acoustic Guitar', 'Fresh', 0.7),
        'afternoon': ('Piano', 'Balanced', 0.8),
        'evening': ('Saxophone', 'Mellow', 0.6),
        'night': ('Synth Pads', 'Dreamy', 0.5)
    }
    
    if time_of_day in time_mappings:
        instruments.append(time_mappings[time_of_day][0])
        moods.append(time_mappings[time_of_day][1])
        config_updates['brightness'] = time_mappings[time_of_day][2]
    
    # Motion mapping
    if motion > 0.5:  # High motion
        genres.append('Drum & Bass')
        config_updates['bpm'] = min(180, config_updates.get('bpm', 120) + 20)
        config_updates['density'] = 0.8
    elif motion > 0.2:  # Medium motion
        genres.append('Funk')
        config_updates['bpm'] = 120
        config_updates['density'] = 0.6
    else:  # Low motion
        genres.append('Ambient')
        config_updates['bpm'] = 80
        config_updates['density'] = 0.4
    
    # Color mood mapping
    color_moods = {
        'red': 'Passionate',
        'blue': 'Calm',
        'green': 'Organic',
        'yellow': 'Happy',
        'black': 'Dark',
        'white': 'Pure',
        'neutral': 'Balanced'
    }
    
    for color in colors:
        if color in color_moods:
            moods.append(color_moods[color])
    
    # Scale selection based on mood
    positive_moods = ['Happy', 'Upbeat', 'Bright', 'Pure']
    negative_moods = ['Melancholic', 'Ominous', 'Dark', 'Dreamy']
    
    positive_count = sum(1 for mood in moods if mood in positive_moods)
    negative_count = sum(1 for mood in moods if mood in negative_moods)
    
    if positive_count > negative_count:
        config_updates['scale'] = types.Scale.C_MAJOR_A_MINOR
    elif negative_count > positive_count:
        config_updates['scale'] = types.Scale.E_FLAT_MAJOR_C_MINOR
    else:
        config_updates['scale'] = types.Scale.G_MAJOR_E_MINOR
    
    # Create weighted prompts
    weighted_prompts = []
    
    # Add primary genre with highest weight
    if genres:
        weighted_prompts.append(
            types.WeightedPrompt(text=f"{genres[0]} genre", weight=1.0)
        )
    
    # Add instruments with moderate weights
    unique_instruments = list(set(instruments))
    for i, inst in enumerate(unique_instruments[:3]):  # Limit to top 3 instruments
        weight = 0.7 - (i * 0.1)  # First instrument gets higher weight
        weighted_prompts.append(types.WeightedPrompt(text=inst, weight=max(0.3, weight)))
    
    # Add moods
    unique_moods = list(set(moods))
    for mood in unique_moods[:2]:  # Limit to top 2 moods
        weighted_prompts.append(types.WeightedPrompt(text=mood, weight=0.5))
    
    # Add weather influence if significant
    if weather in ['rainy', 'sunny', 'fog']:
        weighted_prompts.append(
            types.WeightedPrompt(text=f"{weather} atmosphere", weight=0.4)
        )
    
    return weighted_prompts, config_updates

# ---------- Music AI Setup ----------

BUFFER_SECONDS = 1
CHUNK = 4200
FORMAT = pyaudio.paInt16
CHANNELS = 2
MODEL = 'models/lyria-realtime-exp'
OUTPUT_RATE = 48000

load_dotenv()
api_key = os.environ.get("LYRIA_API_KEY")
if api_key is None:
    print("Please enter your API key")
    api_key = input("API Key: ").strip()

client = genai.Client(
    api_key=api_key,
    http_options={'api_version': 'v1alpha',},
)

# ---------- Main Async Function ----------

async def main():
    # Video setup
    if not os.path.exists('resnet18_places365.pth.tar') or not os.path.exists('categories_places365.txt'):
        print("Missing model or categories file.")
        return

    classes = load_categories()
    places_model = load_places_model()
    yolo_model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture('http://192.168.2.106:8080/video')

    prev_gray = None

    # Music AI setup
    p = pyaudio.PyAudio()
    config = types.LiveMusicGenerationConfig()
    last_prompt = None
    last_prompt_time = 0

    async with client.aio.live.music.connect(model=MODEL) as session:
        print("Connected to music session.")

        async def receive():
            chunks_count = 0
            output_stream = p.open(
                format=FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True, frames_per_buffer=CHUNK)
            async for message in session.receive():
                chunks_count += 1
                if chunks_count == 1:
                    await asyncio.sleep(BUFFER_SECONDS)
                if message.server_content:
                    audio_data = message.server_content.audio_chunks[0].data
                    output_stream.write(audio_data)
                elif message.filtered_prompt:
                    print("Prompt was filtered out: ", message.filtered_prompt)
                else:
                    print("Unknown error occurred with message: ", message)
                await asyncio.sleep(10**-12)

        async def video_music_loop():
            nonlocal prev_gray, last_prompt, last_prompt_time
            print("Press 'q' in the video window to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_resized = cv2.resize(frame, (640, 480))
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

                input_tensor = preprocess_image(frame_resized)
                with torch.no_grad():
                    probs = F.softmax(places_model(input_tensor), 1)[0]
                top_probs, top_idxs = torch.topk(probs, 3)
                top_scenes = [(classes[i], float(p)) for i, p in zip(top_idxs, top_probs)]

                brightness = get_brightness(frame_resized)
                weather = simulate_weather(frame_resized)
                time_of_day = get_time_of_day()
                motion = detect_motion(gray, prev_gray)
                prev_gray = gray
                objects = detect_objects(yolo_model, frame_resized)
                colors = get_dominant_colors(frame_resized)

                object_names = ', '.join([obj['name'] for obj in objects]) if objects else 'None'
                info = [
                    f"Scene 1: {top_scenes[0][0]} ({top_scenes[0][1]*100:.1f}%)",
                    f"Scene 2: {top_scenes[1][0]} ({top_scenes[1][1]*100:.1f}%)",
                    f"Scene 3: {top_scenes[2][0]} ({top_scenes[2][1]*100:.1f}%)",
                    f"Objects: {object_names}",
                    f"Lighting: {brightness}",
                    f"Weather: {weather}",
                    f"Time: {time_of_day}",
                    f"Motion: {motion:.2f}",
                    f"Colors: {' '.join(colors)}"
                ]

                y = 30
                for line in info:
                    cv2.putText(frame_resized, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y += 25

                await asyncio.to_thread(cv2.imshow, "Live Analyzer", frame_resized)
                if await asyncio.to_thread(lambda: cv2.waitKey(1) & 0xFF == ord('q')):
                    break

                now = time.time()
                if now - last_prompt_time >= 5:
                    # Generate music prompts using our enhanced mapping
                    weighted_prompts, config_updates = map_to_music_prompt(
                        top_scenes, objects, brightness, weather, time_of_day, motion, colors
                    )
                    
                    # Update music configuration if needed
                    if config_updates:
                        if 'bpm' in config_updates:
                            config.bpm = config_updates['bpm']
                        if 'brightness' in config_updates:
                            config.brightness = config_updates['brightness']
                        if 'density' in config_updates:
                            config.density = config_updates['density']
                        if 'scale' in config_updates:
                            config.scale = config_updates['scale']
                        
                        print(f"Updating config: BPM={config.bpm}, Brightness={config.brightness}, Density={config.density}, Scale={config.scale.name}")
                        await session.set_music_generation_config(config=config)
                    
                    # Create prompt string for display
                    prompt_str = ", ".join([f"{p.text}:{p.weight}" for p in weighted_prompts])
                    print(f"Generated music prompt: {prompt_str}")

                    if weighted_prompts:
                        await session.set_weighted_prompts(prompts=weighted_prompts)
                        last_prompt_time = now

            cap.release()
            cv2.destroyAllWindows()

        # Initial configuration
        config.bpm = 120
        config.scale = types.Scale.A_FLAT_MAJOR_F_MINOR
        config.brightness = 0.7
        config.density = 0.6
        config.guidance = 4.0
        
        print(f"Setting initial config: BPM={config.bpm}, Scale={config.scale.name}")
        await session.set_music_generation_config(config=config)
        await session.play()

        await asyncio.gather(receive(), video_music_loop())

    p.terminate()

if __name__ == "__main__":
    asyncio.run(main())