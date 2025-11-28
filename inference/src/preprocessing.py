import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """Simple face detector using OpenCV."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        # Use OpenCV's DNN face detector
        self.face_cascade = None
        try:
            # Try to initialize the face cascade - suppress linter warning with getattr
            haarcascades_attr = getattr(cv2, 'data', None)
            if haarcascades_attr:
                cascade_file = getattr(haarcascades_attr, 'haarcascades', '')
                if cascade_file:
                    cascade_path = f"{cascade_file}haarcascade_frontalface_default.xml"
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            logger.warning(f"Could not load Haar cascade: {e}")
        
        logger.info("Face detector initialized")
    
    def process_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
        """Detect and extract faces from a frame."""
        try:
            # Check if face cascade is available
            if self.face_cascade is None or self.face_cascade.empty():
                # Fallback: return the entire frame as a "face"
                resized_frame = cv2.resize(frame, target_size)
                return [resized_frame]
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Extract face regions
            face_images = []
            for (x, y, w, h) in faces:
                # Expand bounding box slightly
                expand = 0.1
                x1 = max(0, int(x - w * expand))
                y1 = max(0, int(y - h * expand))
                x2 = min(frame.shape[1], int(x + w * (1 + expand)))
                y2 = min(frame.shape[0], int(y + h * (1 + expand)))
                
                # Extract face
                face = frame[y1:y2, x1:x2]
                
                # Resize to target size
                face = cv2.resize(face, target_size)
                face_images.append(face)
            
            return face_images
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            # Fallback: return the entire frame as a "face"
            try:
                resized_frame = cv2.resize(frame, target_size)
                return [resized_frame]
            except:
                return []


class VideoProcessor:
    """Video processor for extracting frames and detecting faces."""
    
    def __init__(
        self,
        face_detector: FaceDetector,
        target_fps: float = 2.0,
        max_frames: int = 16,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.face_detector = face_detector
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.target_size = target_size
    
    def process_video(self, video_path: str) -> Tuple[List[np.ndarray], List[int]]:
        """Process video and extract face sequences."""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame sampling interval
            frame_interval = max(1, int(fps / self.target_fps))
            
            face_sequences = []
            frame_indices = []
            
            frame_count = 0
            sampled_count = 0
            
            while cap.isOpened() and sampled_count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames at target FPS
                if frame_count % frame_interval == 0:
                    # Detect faces in frame
                    faces = self.face_detector.process_frame(frame, self.target_size)
                    
                    if faces:
                        # Use the largest face
                        largest_face = max(faces, key=lambda f: f.shape[0] * f.shape[1])
                        face_sequences.append(largest_face)
                        frame_indices.append(frame_count)
                        sampled_count += 1
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Processed video: {len(face_sequences)} faces extracted from {sampled_count} frames")
            return face_sequences, frame_indices
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return [], []


class DataAugmentation:
    """Simple data augmentation utilities."""
    
    @staticmethod
    def get_validation_transforms():
        """Get transforms for validation (minimal augmentation)."""
        # For simplicity, we're returning None here
        # In a real implementation, you might use albumentations or torchvision transforms
        return None
    
    @staticmethod
    def get_training_transforms():
        """Get transforms for training (with augmentation)."""
        # For simplicity, we're returning None here
        # In a real implementation, you might use albumentations or torchvision transforms
        return None