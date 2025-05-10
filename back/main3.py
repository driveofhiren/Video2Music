import cv2
import numpy as np
import time
import fluidsynth
from collections import deque
from ultralytics import YOLO
import os

# Configuration
CONFIG = {
    # Audio settings
    'sample_rate': 44100,
    'buffer_size': 1024,
    'bpm': 120,
    'beats_per_bar': 4,
    
    # Video analysis
    'analysis_fps': 15,
    'camera_index': 0,  # Usually 0 for default webcam
    
    # SoundFont paths (download these)
    'soundfonts': {
        'melody': 'flute.sf2',
        'bass': 'bass.sf2',
        'drums': 'drums.sf2',
        'pads': 'strings.sf2'
    },
    
    # Instrument settings
    'instruments': {
        'melody': {'program': 73, 'channel': 0, 'volume': 0.8},
        'bass': {'program': 33, 'channel': 1, 'volume': 0.7},
        'drums': {'program': 0, 'channel': 9, 'volume': 0.9},
        'pads': {'program': 49, 'channel': 2, 'volume': 0.5}
    },
    
    # Musical scales
    'scales': {
        'minor': [0, 2, 3, 5, 7, 8, 10],  # Semitone offsets
        'major': [0, 2, 4, 5, 7, 9, 11],
        'dorian': [0, 2, 3, 5, 7, 9, 10]
    },
    'key': 'C'  # Root note
}

class LiveVideoToMusic:
    def __init__(self):
        # Initialize audio engine
        self.fs = fluidsynth.Synth()
        self.fs.start(driver="coreaudio")  # "alsa" on Linux
        
        # Load SoundFonts
        self.sfids = {}
        for name, sf_path in CONFIG['soundfonts'].items():
            try:
                self.sfids[name] = self.fs.sfload(sf_path)
                self.fs.program_select(
                    CONFIG['instruments'][name]['channel'],
                    self.sfids[name],
                    0,
                    CONFIG['instruments'][name]['program']
                )
            except Exception as e:
                print(f"Error loading {sf_path}: {e}")
                # Fallback to default soundfont
                self.sfids[name] = self.fs.sfload('default.sf2')
        
        # Initialize video
        self.cap = cv2.VideoCapture(CONFIG['camera_index'])
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video source")
        
        # Load object detection model
        self.yolo = YOLO('yolov8n.pt')
        self.prev_frame = None
        
        # Music timing
        self.beat_interval = 60 / CONFIG['bpm']
        self.next_beat = time.time() + self.beat_interval
        self.beat_count = 0
        
        # State tracking
        self.object_history = deque(maxlen=10)
        self.motion_history = deque(maxlen=5)
        self.current_scale = CONFIG['scales']['minor']
        self.root_note = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'].index(CONFIG['key'])
        self.current_chord = self._get_chord(1)  # Start with root chord
        
        # Performance metrics
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.fps = 0

    def run(self):
        print("Starting live video-to-music performance...")
        print(f"Tempo: {CONFIG['bpm']} BPM | Key: {CONFIG['key']} {self._get_scale_name()}")
        
        try:
            while True:
                # 1. Timing and FPS calculation
                self.frame_count += 1
                if time.time() - self.last_fps_update >= 1.0:
                    self.fps = self.frame_count / (time.time() - self.last_fps_update)
                    self.last_fps_update = time.time()
                    self.frame_count = 0
                
                # 2. Capture video frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # 3. Process frame at configured rate
                if self.frame_count % (30 // CONFIG['analysis_fps']) == 0:
                    self._process_frame(frame)
                
                # 4. Generate musical events on beats
                now = time.time()
                if now >= self.next_beat:
                    self._generate_beat()
                    self.next_beat += self.beat_interval
                    self.beat_count += 1
                
                # 5. Display video with overlay
                self._display_frame(frame)
                
                # 6. Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self._cleanup()

    def _process_frame(self, frame):
        # 1. Object detection
        results = self.yolo(frame, verbose=False)[0]
        current_objects = [self.yolo.names[int(cls)] for cls in results.boxes.cls]
        self.object_history.append(current_objects)
        
        # 2. Motion analysis
        motion = 0
        if self.prev_frame is not None:
            gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray_prev, gray_current, None, 
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            motion = np.mean(np.sqrt(np.sum(flow**2, axis=2)))
        self.prev_frame = frame.copy()
        self.motion_history.append(motion)
        
        # 3. Update musical parameters
        self._update_music_params(current_objects, motion)

    def _update_music_params(self, objects, motion):
        # Change scale based on objects detected
        if 'person' in objects:
            self.current_scale = CONFIG['scales']['minor']
        elif 'car' in objects or 'bus' in objects:
            self.current_scale = CONFIG['scales']['dorian']
        else:
            self.current_scale = CONFIG['scales']['major']
        
        # Change chord based on beat position
        chord_degree = (self.beat_count // 2) % 4 + 1  # Cycle through I-IV chords
        self.current_chord = self._get_chord(chord_degree)
        
        # Adjust tempo based on motion
        if len(self.motion_history) > 3:
            avg_motion = np.mean(self.motion_history)
            CONFIG['bpm'] = min(160, max(80, 100 + avg_motion * 40))
            self.beat_interval = 60 / CONFIG['bpm']

    def _generate_beat(self):
        beat_in_bar = self.beat_count % CONFIG['beats_per_bar']
        avg_motion = np.mean(self.motion_history) if self.motion_history else 0
        
        # 1. Drums - rhythm foundation
        self._play_drums(beat_in_bar, avg_motion)
        
        # 2. Bass - on strong beats
        if beat_in_bar % 2 == 0:
            self._play_bass()
        
        # 3. Melody - based on objects
        if len(self.object_history) > 3:
            self._play_melody()
        
        # 4. Pads - harmonic support
        if beat_in_bar == 0:  # Only change pads on downbeat
            self._play_pads()

    def _play_drums(self, beat, motion):
        # Channel 10 is drums in General MIDI
        channel = CONFIG['instruments']['drums']['channel']
        
        # Kick drum on beat 1 and sometimes 3
        if beat == 0 or (beat == 2 and motion > 0.5):
            self.fs.noteon(channel, 36, int(90 + motion * 20))  # Kick
            self.fs.noteoff(channel, 36)
        
        # Snare on beats 2 and 4
        if beat == 1 or beat == 3:
            self.fs.noteon(channel, 38, 80)  # Snare
            self.fs.noteoff(channel, 38)
        
        # Hi-hat based on motion intensity
        velocity = int(70 + motion * 30)
        self.fs.noteon(channel, 42, velocity)  # Closed hi-hat
        self.fs.noteoff(channel, 42)

    def _play_bass(self):
        channel = CONFIG['instruments']['bass']['channel']
        root = self.current_chord[0] - 12  # Play bass octave
        self.fs.noteon(channel, root, 80)
        self.fs.noteoff(channel, root)

    def _play_melody(self):
        channel = CONFIG['instruments']['melody']['channel']
        
        # Get most common object in history
        all_objects = [obj for sublist in self.object_history for obj in sublist]
        if not all_objects:
            return
            
        main_object = max(set(all_objects), key=all_objects.count)
        note = self._object_to_note(main_object)
        
        # Add some variation
        note_variation = random.choice([-1, 0, 0, 1])  # Occasional neighbor tones
        final_note = max(48, min(84, note + note_variation))  # Keep in range
        
        self.fs.noteon(channel, final_note, 90)
        self.fs.noteoff(channel, final_note)

    def _play_pads(self):
        channel = CONFIG['instruments']['pads']['channel']
        
        # First release any sustained notes
        for note in range(48, 72):
            self.fs.noteoff(channel, note)
        
        # Play current chord
        for note in self.current_chord:
            self.fs.noteon(channel, note, 60)
            # Note-offs will be handled on next chord change

    def _object_to_note(self, obj):
        """Map objects to musical notes in current scale"""
        obj_hash = hash(obj) % len(self.current_scale)
        return self.current_scale[obj_hash] + self.root_note + 60  # Middle C octave

    def _get_chord(self, degree):
        """Get chord notes (1-based degree) in current scale"""
        return [
            (self.current_scale[(degree-1+i)%len(self.current_scale)] + self.root_note + 60)
            for i in [0, 2, 4]  # Root, third, fifth
        ]

    def _get_scale_name(self):
        for name, scale in CONFIG['scales'].items():
            if list(scale) == list(self.current_scale):
                return name
        return "custom"

    def _display_frame(self, frame):
        # Add performance overlay
        overlay_text = [
            f"FPS: {self.fps:.1f}",
            f"BPM: {CONFIG['bpm']:.0f}",
            f"Objects: {len(set(sum(self.object_history, [])))}",
            f"Motion: {np.mean(self.motion_history) if self.motion_history else 0:.2f}",
            f"Scale: {CONFIG['key']} {self._get_scale_name()}"
        ]
        
        for i, text in enumerate(overlay_text):
            cv2.putText(
                frame, text, (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        # Visual metronome
        beat_progress = (self.next_beat - time.time()) / self.beat_interval
        cv2.circle(
            frame, 
            (frame.shape[1] - 30, 30), 
            15, 
            (0, 255 * (1 - beat_progress), 255 * beat_progress), 
            -1
        )
        
        cv2.imshow('Live Video-to-Music', frame)

    def _cleanup(self):
        print("\nCleaning up resources...")
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Release all notes
        for instr in CONFIG['instruments'].values():
            for note in range(128):
                self.fs.noteoff(instr['channel'], note)
        
        self.fs.delete()
        print("Performance ended")

if __name__ == "__main__":
    # Check for required files
    required_sf = set(CONFIG['soundfonts'].values())
    missing = [sf for sf in required_sf if not os.path.exists(sf)]
    
    if missing:
        print(f"Missing SoundFont files: {missing}")
        print("Please download and place in project folder:")
        print("1. Try https://freepats.zenvoid.org/")
        print("2. Or search for 'free soundfonts'")
    else:
        performer = LiveVideoToMusic()
        performer.run()