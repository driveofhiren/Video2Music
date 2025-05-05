import pygame
import os
import threading
import time
from mingus.core import notes, scales
from random import choice

pygame.init()
pygame.mixer.init(frequency=44100)

class AdaptiveMusicEngine:
    def __init__(self, audio_dir="assets"):
        self.audio_dir = audio_dir
        self.current_track = None
        self.current_path = None
        self.fade_time = 2000  # ms
        self.channel = pygame.mixer.Channel(0)
        self.lock = threading.Lock()
        self.melody_thread = None

    def update(self, scene_data):
        """Decide and update background loop based on scene"""
        filename = self._generate_filename(scene_data)
        path = os.path.join(self.audio_dir, filename)
        
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            return
        
        with self.lock:
            if self.current_path == path:
                return  # same track, skip
            print(f"[INFO] Switching to {filename}")
            self._crossfade_to(path)
            self.current_path = path

        # Start melody generator based on time of day
        if self.melody_thread is None or not self.melody_thread.is_alive():
            self.melody_thread = threading.Thread(target=self._play_generated_melody, args=(scene_data,), daemon=True)
            self.melody_thread.start()

    def _generate_filename(self, data):
        # Example: indoor_rainy_morning.wav
        return f"{data['scene']}_{data['weather']}_{data['time_of_day']}.wav"

    def _crossfade_to(self, path):
        """Load and fade in new track"""
        new_track = pygame.mixer.Sound(path)
        self.channel.fadeout(self.fade_time)
        time.sleep(self.fade_time / 1000.0)
        self.channel.play(new_track, loops=-1, fade_ms=self.fade_time)

    def _play_generated_melody(self, data):
        """Generate a melody in key and print as notes (optional MIDI later)"""
        key = 'A' if data['weather'] == 'rainy' else 'C'
        scale_type = 'minor' if data['weather'] == 'rainy' else 'major'
        scale = scales.Aeolian(key) if scale_type == 'minor' else scales.Major(key)
        notes_list = scale.ascending()

        print(f"[Melody] Playing {key} {scale_type} scale melody...")

        for _ in range(8):
            note = choice(notes_list)
            print(f"🎶 Note: {note}")
            pygame.mixer.Sound(os.path.join("assets", "melody_notes", f"{note}.wav")).play()
            time.sleep(0.5)


def mock_scene_data():
    """Fake output from AdvancedSceneAnalyzer"""
    return {
        "scene": choice(["indoor", "urban", "nature"]),
        "weather": choice(["sunny", "rainy", "cloudy"]),
        "time_of_day": choice(["morning", "afternoon", "evening", "night"]),
        "movement": round(choice([1.2, 3.4, 7.8]), 2),
        "objects": ["person"],
        "color_palette": ["#aaa", "#333"]
    }

def main():
    music_engine = AdaptiveMusicEngine()

    try:
        while True:
            scene_data = mock_scene_data()
            print(f"\nScene Analysis: {scene_data}")
            music_engine.update(scene_data)
            time.sleep(10)  # simulate new frame every 10s
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        pygame.quit()

if __name__ == "__main__":
    main()
