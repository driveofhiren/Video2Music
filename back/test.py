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
from sentence_transformers import SentenceTransformer, util

import pyaudio
from google import genai
from google.genai import types
from dotenv import load_dotenv


model_st = SentenceTransformer('all-MiniLM-L6-v2')

# Lyria RealTime genre and mood lists (sample, expand as needed)
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
    "Piano Ballad", "Polka", "Post-Punk", "Psytrance", "R&B",
    "Reggae", "Reggaeton", "Renaissance Music", "Salsa", "Shoegaze",
    "Ska", "Surf Rock", "Synthpop", "Techno", "Trance",
    "Trap Beat", "Trip Hop", "Vaporwave", "Witch house"
]

MOODS = [
    "Acoustic Instruments", "Ambient", "Bright Tones", "Chill", "Crunchy Distortion",
    "Danceable", "Dreamy", "Echo", "Emotional", "Ethereal Ambience",
    "Experimental", "Fat Beats", "Funky", "Glitchy Effects", "Huge Drop",
    "Live Performance", "Lo-fi", "Ominous Drone", "Psychedelic", "Rich Orchestration",
    "Saturated Tones", "Subdued Melody", "Sustained Chords", "Swirling Phasers",
    "Tight Groove", "Unsettling", "Upbeat", "Virtuoso", "Weird Noises"
]

# Precompute embeddings for genres and moods (run once)
genre_embs = model_st.encode(GENRES, convert_to_tensor=True)
mood_embs = model_st.encode(MOODS, convert_to_tensor=True)
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
    instruments = []
    prompts = []
    config_updates = {}

    # --- 1. Use sentence-transformers for genre + mood ---
    scene_text = ', '.join([s[0].lower() for s in top_scenes])
    scene_emb = model_st.encode(scene_text, convert_to_tensor=True)
    genre_scores = util.cos_sim(scene_emb, genre_embs)
    mood_scores = util.cos_sim(scene_emb, mood_embs)
    best_genre = GENRES[genre_scores.argmax()]
    best_mood = MOODS[mood_scores.argmax()]

    prompts.append(types.WeightedPrompt(text=f"{best_genre} genre", weight=1.0))
    prompts.append(types.WeightedPrompt(text=best_mood, weight=0.7))

    # --- 2. Add object-derived instruments (if you want to keep that) ---
    object_mappings = {
        'car': 'Dirty Synths',
        'tree': 'Wind Chimes',
        'dog': 'Whistles',
        'cat': 'Purring Sounds',
        'computer': 'Chiptune',
        'phone': 'Electronic Beeps',
        'book': 'Page Turning Sounds',
        'chair': 'Wooden Percussion',
        'cup': 'Glass Harmonica',
    }
    for obj in objects:
        name = obj['name'].lower()
        for key in object_mappings:
            if key in name:
                instruments.append(object_mappings[key])

    for i, inst in enumerate(list(set(instruments))[:3]):
        weight = 0.7 - (i * 0.1)
        prompts.append(types.WeightedPrompt(text=inst, weight=weight))

    # --- 3. Direct prompt embedding from visual context ---
    prompts.append(types.WeightedPrompt(text=f"{brightness} lighting", weight=0.5))
    prompts.append(types.WeightedPrompt(text=f"{weather} weather", weight=0.5))
    prompts.append(types.WeightedPrompt(text=f"{time_of_day} time", weight=0.5))
    
    if motion > 0.5:
        prompts.append(types.WeightedPrompt(text="high motion", weight=0.6))
        config_updates['bpm'] = 160
        config_updates['density'] = 0.9
    elif motion > 0.2:
        prompts.append(types.WeightedPrompt(text="medium motion", weight=0.5))
        config_updates['bpm'] = 120
        config_updates['density'] = 0.6
    else:
        prompts.append(types.WeightedPrompt(text="low motion", weight=0.4))
        config_updates['bpm'] = 80
        config_updates['density'] = 0.3

    # --- 4. Add dominant colors as descriptive tags ---
    color_desc = ", ".join(colors)
    prompts.append(types.WeightedPrompt(text=f"color palette: {color_desc}", weight=0.4))

    # --- 5. Scale selection based on mood polarity ---
    positive_moods = ['Happy', 'Upbeat', 'Bright', 'Pure']
    negative_moods = ['Melancholic', 'Ominous', 'Dark', 'Dreamy']
    
    pos_count = sum(1 for m in [best_mood] if m in positive_moods)
    neg_count = sum(1 for m in [best_mood] if m in negative_moods)

    if pos_count > neg_count:
        config_updates['scale'] = types.Scale.C_MAJOR_A_MINOR
    elif neg_count > pos_count:
        config_updates['scale'] = types.Scale.E_FLAT_MAJOR_C_MINOR
    else:
        config_updates['scale'] = types.Scale.G_MAJOR_E_MINOR

    return prompts, config_updates

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
        config.guidance = 6.0
        
        print(f"Setting initial config: BPM={config.bpm}, Scale={config.scale.name}")
        await session.set_music_generation_config(config=config)
        await session.play()

        await asyncio.gather(receive(), video_music_loop())

    p.terminate()

if __name__ == "__main__":
    asyncio.run(main())