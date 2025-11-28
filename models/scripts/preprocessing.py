import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import mediapipe as mp
import face_recognition
from pathlib import Path
import json
import logging
from typing import List, Tuple, Dict, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
import os
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection and alignment utility using MediaPipe and face_recognition."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=confidence_threshold
        )
    
    def detect_faces_mediapipe(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe."""
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        faces = []
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                faces.append((x, y, x + w, y + h))
        
        return faces
    
    def detect_faces_dlib(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using face_recognition (dlib backend)."""
        face_locations = face_recognition.face_locations(image)
        faces = []
        
        for (top, right, bottom, left) in face_locations:
            faces.append((left, top, right, bottom))
        
        return faces
    
    def extract_face(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        target_size: Tuple[int, int] = (224, 224),
        margin: float = 0.2
    ) -> Optional[np.ndarray]:
        """Extract and align face from image."""
        x1, y1, x2, y2 = bbox
        
        # Add margin
        margin_x = int((x2 - x1) * margin)
        margin_y = int((y2 - y1) * margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(image.shape[1], x2 + margin_x)
        y2 = min(image.shape[0], y2 + margin_y)
        
        # Extract face region
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        # Resize to target size
        face_resized = cv2.resize(face, target_size)
        
        return face_resized
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        target_size: Tuple[int, int] = (224, 224)
    ) -> List[np.ndarray]:
        """Process a single frame and extract all faces."""
        faces = []
        
        # Try MediaPipe first
        face_bboxes = self.detect_faces_mediapipe(frame)
        
        # Fallback to dlib if no faces found
        if not face_bboxes:
            face_bboxes = self.detect_faces_dlib(frame)
        
        for bbox in face_bboxes:
            face = self.extract_face(frame, bbox, target_size)
            if face is not None:
                faces.append(face)
        
        return faces


class VideoProcessor:
    """Video processing utilities for extracting frames and faces."""
    
    def __init__(
        self, 
        face_detector: FaceDetector,
        target_fps: float = 2.0,
        max_frames: int = 50,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.face_detector = face_detector
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.target_size = target_size
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video at target FPS."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            fps = 30  # Default FPS
        
        # Calculate frame interval
        frame_interval = max(1, int(fps / self.target_fps))
        
        frames = []
        frame_count = 0
        
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def process_video(self, video_path: str) -> Tuple[List[np.ndarray], List[int]]:
        """Process video and extract face sequences."""
        frames = self.extract_frames(video_path)
        face_sequences = []
        valid_frame_indices = []
        
        for i, frame in enumerate(frames):
            faces = self.face_detector.process_frame(frame, self.target_size)
            
            if faces:
                # Use the largest face if multiple faces detected
                largest_face = max(faces, key=lambda f: f.shape[0] * f.shape[1])
                face_sequences.append(largest_face)
                valid_frame_indices.append(i)
        
        if not face_sequences:
            raise ValueError(f"No faces detected in video: {video_path}")
        
        logger.info(f"Processed {len(face_sequences)} faces from {video_path}")
        return face_sequences, valid_frame_indices


class DeepfakeDataset(Dataset):
    """Dataset class for deepfake detection."""
    
    def __init__(
        self,
        data_list: List[Dict],
        transform: Optional[A.Compose] = None,
        sequence_length: int = 16,
        is_training: bool = True
    ):
        self.data_list = data_list
        self.transform = transform
        self.sequence_length = sequence_length
        self.is_training = is_training
        
        # Initialize processors
        self.face_detector = FaceDetector()
        self.video_processor = VideoProcessor(
            face_detector=self.face_detector,
            max_frames=sequence_length
        )
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data_list[idx]
        file_path = item['path']
        label = item['label']  # 0: authentic, 1: deepfake
        
        try:
            # Check if it's an image or video file
            file_extension = Path(file_path).suffix.lower()
            is_image = file_extension in ['.jpg', '.jpeg', '.png']
            
            if is_image:
                # Process image file
                face_sequence = self._process_image(file_path)
            else:
                # Process video file
                face_sequence, frame_indices = self.video_processor.process_video(file_path)
            
            # Pad or truncate sequence
            face_sequence = self._prepare_sequence(face_sequence)
            
            # Apply transformations
            if self.transform:
                face_sequence = [self.transform(image=face)['image'] for face in face_sequence]
            else:
                # Convert to tensor
                face_sequence = [torch.from_numpy(face.transpose(2, 0, 1)).float() / 255.0 
                               for face in face_sequence]
            
            # Stack into sequence tensor
            sequence_tensor = torch.stack(face_sequence)
            
            return {
                'frames': sequence_tensor,
                'label': torch.tensor(label, dtype=torch.long),
                'video_path': file_path,
                'frame_indices': torch.zeros(self.sequence_length, dtype=torch.long) if is_image else torch.tensor(frame_indices[:len(face_sequence)], dtype=torch.long)
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            # Return dummy data for failed samples
            dummy_frames = torch.zeros(self.sequence_length, 3, 224, 224)
            return {
                'frames': dummy_frames,
                'label': torch.tensor(0, dtype=torch.long),
                'video_path': file_path,
                'frame_indices': torch.zeros(self.sequence_length, dtype=torch.long)
            }
    
    def _process_image(self, image_path: str) -> List[np.ndarray]:
        """Process a single image and extract face."""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect and extract faces
        faces = self.face_detector.process_frame(image, (224, 224))
        
        if not faces:
            # If no face detected, use the whole image
            resized_image = cv2.resize(image, (224, 224))
            return [resized_image]
        
        # Use the largest face if multiple faces detected
        largest_face = max(faces, key=lambda f: f.shape[0] * f.shape[1])
        return [largest_face]
    
    def _prepare_sequence(self, face_sequence: List[np.ndarray]) -> List[np.ndarray]:
        """Prepare face sequence with proper length."""
        if len(face_sequence) >= self.sequence_length:
            # Truncate or sample
            if self.is_training:
                # Random sampling during training
                indices = np.random.choice(
                    len(face_sequence), 
                    self.sequence_length, 
                    replace=False
                )
                indices.sort()
            else:
                # Uniform sampling during inference
                indices = np.linspace(
                    0, 
                    len(face_sequence) - 1, 
                    self.sequence_length, 
                    dtype=int
                )
            
            face_sequence = [face_sequence[i] for i in indices]
        else:
            # Pad sequence by repeating last frame
            last_frame = face_sequence[-1] if face_sequence else np.zeros((224, 224, 3), dtype=np.uint8)
            while len(face_sequence) < self.sequence_length:
                face_sequence.append(last_frame.copy())
        
        return face_sequence


class DataAugmentation:
    """Data augmentation for deepfake detection."""
    
    @staticmethod
    def get_training_transforms() -> A.Compose:
        """Get training augmentations."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=20, 
                val_shift_limit=10, 
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.OneOf([
                A.MotionBlur(p=1.0, blur_limit=5),
                A.GaussianBlur(p=1.0, blur_limit=5),
            ], p=0.3),
            A.JpegCompression(quality_lower=70, quality_upper=100, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_validation_transforms() -> A.Compose:
        """Get validation transforms."""
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


class DatasetBuilder:
    """Build datasets from various deepfake datasets."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
    
    def build_archive_dataset(self, split: str = 'train') -> List[Dict]:
        """Build archive dataset list from the archive/Dataset directory."""
        data_list = []
        # Use absolute path for archive dataset
        archive_path = Path("archive/Dataset") / split.capitalize()
        
        if not archive_path.exists():
            logger.warning(f"Archive path does not exist: {archive_path}")
            return data_list
        
        # Process Real directory (authentic)
        real_path = archive_path / "Real"
        if real_path.exists():
            # Handle multiple image extensions
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for image_file in real_path.glob(ext):
                    data_list.append({
                        'path': str(image_file.absolute()),
                        'label': 0,  # 0 for authentic
                        'dataset': 'Archive',
                        'split': split,
                        'type': 'real'
                    })
        
        # Process Fake directory (deepfake)
        fake_path = archive_path / "Fake"
        if fake_path.exists():
            # Handle multiple image extensions
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for image_file in fake_path.glob(ext):
                    data_list.append({
                        'path': str(image_file.absolute()),
                        'label': 1,  # 1 for deepfake
                        'dataset': 'Archive',
                        'split': split,
                        'type': 'fake'
                    })
        
        logger.info(f"Built Archive {split} dataset: {len(data_list)} samples")
        return data_list
    
    def build_dfdc_dataset(self, split: str = 'train') -> List[Dict]:
        """Build DFDC dataset list."""
        data_list = []
        dfdc_path = self.data_root / 'DFDC' / split
        
        if not dfdc_path.exists():
            logger.warning(f"DFDC path does not exist: {dfdc_path}")
            return data_list
        
        # Load metadata
        metadata_file = dfdc_path / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for video_name, info in metadata.items():
                video_path = dfdc_path / video_name
                if video_path.exists():
                    label = 1 if info['label'] == 'FAKE' else 0
                    data_list.append({
                        'path': str(video_path),
                        'label': label,
                        'dataset': 'DFDC',
                        'split': split
                    })
        
        logger.info(f"Built DFDC {split} dataset: {len(data_list)} samples")
        return data_list
    
    def build_faceforensics_dataset(self, split: str = 'train') -> List[Dict]:
        """Build FaceForensics++ dataset list."""
        data_list = []
        ff_path = self.data_root / 'FaceForensics++' / split
        
        if not ff_path.exists():
            logger.warning(f"FaceForensics++ path does not exist: {ff_path}")
            return data_list
        
        # Original videos (authentic)
        original_path = ff_path / 'original_sequences' / 'youtube' / 'c23' / 'videos'
        if original_path.exists():
            for video_file in original_path.glob('*.mp4'):
                data_list.append({
                    'path': str(video_file),
                    'label': 0,
                    'dataset': 'FaceForensics++',
                    'split': split,
                    'manipulation': 'original'
                })
        
        # Manipulated videos (deepfake)
        manipulations = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        for manipulation in manipulations:
            manip_path = ff_path / 'manipulated_sequences' / manipulation / 'c23' / 'videos'
            if manip_path.exists():
                for video_file in manip_path.glob('*.mp4'):
                    data_list.append({
                        'path': str(video_file),
                        'label': 1,
                        'dataset': 'FaceForensics++',
                        'split': split,
                        'manipulation': manipulation
                    })
        
        logger.info(f"Built FaceForensics++ {split} dataset: {len(data_list)} samples")
        return data_list
    
    def build_celebdf_dataset(self, split: str = 'train') -> List[Dict]:
        """Build Celeb-DF dataset list."""
        data_list = []
        celebdf_path = self.data_root / 'Celeb-DF-v2' / split
        
        if not celebdf_path.exists():
            logger.warning(f"Celeb-DF path does not exist: {celebdf_path}")
            return data_list
        
        # Real videos
        real_path = celebdf_path / 'Celeb-real'
        if real_path.exists():
            for video_file in real_path.glob('*.mp4'):
                data_list.append({
                    'path': str(video_file),
                    'label': 0,
                    'dataset': 'Celeb-DF',
                    'split': split,
                    'manipulation': 'real'
                })
        
        # Synthesis videos
        synthesis_path = celebdf_path / 'Celeb-synthesis'
        if synthesis_path.exists():
            for video_file in synthesis_path.glob('*.mp4'):
                data_list.append({
                    'path': str(video_file),
                    'label': 1,
                    'dataset': 'Celeb-DF',
                    'split': split,
                    'manipulation': 'synthesis'
                })
        
        logger.info(f"Built Celeb-DF {split} dataset: {len(data_list)} samples")
        return data_list
    
    def build_combined_dataset(
        self, 
        split: str = 'train',
        include_datasets: List[str] = ['DFDC', 'FaceForensics++', 'Celeb-DF']
    ) -> List[Dict]:
        """Build combined dataset from multiple sources."""
        combined_data = []
        
        if 'DFDC' in include_datasets:
            combined_data.extend(self.build_dfdc_dataset(split))
        
        if 'FaceForensics++' in include_datasets:
            combined_data.extend(self.build_faceforensics_dataset(split))
        
        if 'Celeb-DF' in include_datasets:
            combined_data.extend(self.build_celebdf_dataset(split))
        
        # Add archive dataset
        if 'Archive' in include_datasets:
            combined_data.extend(self.build_archive_dataset(split))
        
        # Shuffle the combined dataset
        np.random.shuffle(combined_data)
        
        logger.info(f"Built combined {split} dataset: {len(combined_data)} samples")
        return combined_data
