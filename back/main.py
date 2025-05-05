import os
import cv2
import numpy as np
import logging
from datetime import datetime
from ultralytics import YOLO
import time
import random
import pygame
import threading


# Suppress OpenCV logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedSceneAnalyzer:
    def __init__(self):
        """Initialize YOLOv8 model and supporting elements"""
        try:
            self.model = YOLO('yolov8n.pt')  # Load YOLOv8 model
            self.weather_cache = {"time": 0, "type": "unknown"}
            self.prev_frame = None
            logger.info("YOLOv8 model initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def analyze_frame(self, frame):
        """Run full scene analysis"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            return {
                "objects": self._detect_objects(rgb),
                "scene": self._classify_scene_color(rgb),
                "weather": self._get_weather_fallback(),
                "time_of_day": self._get_time_of_day(),
                "color_palette": self._get_dominant_colors(frame),
                "movement": self._detect_movement(frame)
            }
        except Exception as e:
            logger.warning(f"Analysis error: {e}")
            return self._get_fallback_analysis()

    def _detect_objects(self, frame):
        """Use YOLOv8 to detect all visible objects"""
        results = self.model(frame, verbose=False)[0]
        return list(set([self.model.names[int(cls)] for cls in results.boxes.cls]))

    def _classify_scene_color(self, rgb_frame):
        """Simple scene type via saturation and hue"""
        hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_hue = np.mean(hsv[:, :, 0])

        if avg_saturation < 50:
            return "indoor"
        elif avg_hue < 30:
            return "nature"
        return "urban"

    def _get_weather_fallback(self):
        """Simulate weather every hour"""
        if time.time() - self.weather_cache["time"] > 3600:
            self.weather_cache = {
                "type": random.choice(["sunny", "cloudy", "rainy"]),
                "time": time.time()
            }
        return self.weather_cache["type"]

    def _get_time_of_day(self):
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        return "night"

    def _get_dominant_colors(self, frame, n_colors=3):
        """Extract top colors using K-means"""
        pixels = frame.reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, _, centers = cv2.kmeans(
            pixels.astype(np.float32), n_colors, None,
            criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return [f"#{int(c[2]):02x}{int(c[1]):02x}{int(c[0]):02x}" for c in centers]

    def _detect_movement(self, frame):
        """Detect motion between frames"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0

        flow = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray
        return np.mean(flow)

    def _get_fallback_analysis(self):
        """Defaults on error"""
        return {
            "objects": [],
            "scene": "unknown",
            "weather": "unknown",
            "time_of_day": self._get_time_of_day(),
            "color_palette": ["#000000"],
            "movement": 0
        }


def connect_camera():
    """Connect to IP camera or fallback to webcam"""
    sources = ['http://192.168.2.106:8080/video', 0, 1, 2, 3]
    for source in sources:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            logger.info(f"Camera connected: {source}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
    raise RuntimeError("No working camera source found.")



class AdaptiveMusicEngine:
    def __init__(self, audio_dir="assets"):
        pygame.init()
        pygame.mixer.init()
        self.audio_dir = audio_dir
        self.current_path = None
        self.channel = pygame.mixer.Channel(0)
        self.fade_time = 2000
        self.lock = threading.Lock()

    def _generate_filename(self, data):
        return f"{data['scene']}_{data['weather']}_{data['time_of_day']}.wav"

    def update(self, data):
        filename = self._generate_filename(data)
        path = os.path.join(self.audio_dir, filename)

        if not os.path.exists(path):
            logger.warning(f"Audio file not found: {path}")
            return

        with self.lock:
            if self.current_path == path:
                return
            self.current_path = path
            self._crossfade(path)

    def _crossfade(self, path):
        sound = pygame.mixer.Sound(path)
        self.channel.fadeout(self.fade_time)
        pygame.time.wait(self.fade_time)
        self.channel.play(sound, loops=-1, fade_ms=self.fade_time)



def main():
    cap = None
    try:
        cap = connect_camera()
        analyzer = AdvancedSceneAnalyzer()
        music_engine = AdaptiveMusicEngine()  # <-- initialize music engine

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                break

            analysis = analyzer.analyze_frame(frame)

            # Log and update audio
            logger.info(f"""
                Objects: {analysis['objects']}
                Scene: {analysis['scene']}
                Weather: {analysis['weather']}
                Time: {analysis['time_of_day']}
                Colors: {analysis['color_palette']}
                Movement: {analysis['movement']:.2f}
            """)
            music_engine.update(analysis)  # <-- apply audio changes

            # Display
            cv2.putText(frame, f"Objects: {', '.join(analysis['objects'])}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Scene Analysis', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()
