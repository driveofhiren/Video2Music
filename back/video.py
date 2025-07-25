import torch
import torchvision.models as models
import torch.nn.functional as F
import cv2
import numpy as np
import os
from datetime import datetime
from torchvision import transforms
from sklearn.cluster import KMeans
from ultralytics import YOLO

# ---------- Category Loading ----------
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

# ---------- Contextual Analysis ----------
def get_brightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    if brightness < 50:
        return "dark"
    elif brightness < 150:
        return "dim"
    return "bright"

def get_dominant_colors(frame, k=3):
    data = frame.reshape((-1, 3)).astype(np.float32)
    kmeans = KMeans(n_clusters=k, n_init=1)
    kmeans.fit(data)
    colors = kmeans.cluster_centers_.astype(int)
    return [f"#{c[2]:02x}{c[1]:02x}{c[0]:02x}" for c in colors]

def simulate_weather(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = np.mean(hsv[:, :, 0])
    brightness = np.mean(hsv[:, :, 2])
    if brightness < 60:
        return "rainy"
    elif hue > 90 and brightness > 160:
        return "sunny"
    elif 50 < hue < 90:
        return "cloudy"
    else:
        return "unclear"

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
        x1, y1, x2, y2 = map(int, box[:4])
        detected_objects.append({
            'name': names[int(cls)],
            'position': ((x1 + x2) // 2, (y1 + y2) // 2),
            'bbox': (x1, y1, x2, y2),
            'confidence': float(conf)
        })
    return detected_objects

# ---------- Drawing ----------
def draw_object_positions(frame, objects):
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        center_x, center_y = obj['position']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        label = f"{obj['name']} ({center_x},{center_y})"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

def draw_overlay(frame, lines):
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25
    return frame

# ---------- Main ----------
def main():
    if not os.path.exists('resnet18_places365.pth.tar') or not os.path.exists('categories_places365.txt'):
        print("Missing model or categories file.")
        return

    classes = load_categories()
    places_model = load_places_model()
    yolo_model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture('http://192.168.2.106:8080/video')  # Your webcam or IP camera stream

    prev_gray = None

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Scene classification
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

        frame_with_positions = draw_object_positions(frame_resized.copy(), objects)

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

        draw_overlay(frame_with_positions, info)
        cv2.imshow("Live Analyzer", frame_with_positions)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
