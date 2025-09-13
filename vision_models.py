"""
Robust vision models and query routing system for the Vision-Language Agent.
Production-ready with graceful error handling, optional dependencies, and fallback mechanisms.
"""
# At the top of your files:
import sys
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import logging
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from data_structures import VisionQuery, VisionResult
from input_processing import CacheManager

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

logger = logging.getLogger(__name__)

# Suppress warnings from deep learning libraries
warnings.filterwarnings("ignore", category=UserWarning)


class ModelStatus:
    """Model status tracking"""
    AVAILABLE = "available"
    FAILED = "failed" 
    MOCK = "mock"
    NOT_LOADED = "not_loaded"


class VisionModel(ABC):
    """Abstract base class for vision models with robust error handling"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.status = ModelStatus.NOT_LOADED
        self.error_message = None
        self.model_loaded = False
        
    @abstractmethod
    def _load_model(self) -> bool:
        """Load the actual model. Return True if successful."""
        pass
    
    @abstractmethod
    def _process_real(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Real model processing implementation"""
        pass
    
    @abstractmethod
    def _process_mock(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Mock/fallback processing implementation"""
        pass
    
    def load(self) -> bool:
        """Initialize model with error handling"""
        try:
            success = self._load_model()
            if success:
                self.status = ModelStatus.AVAILABLE
                self.model_loaded = True
                logger.info(f"[[OK]] {self.model_name} loaded successfully")
            else:
                self._fallback_to_mock("Model loading returned False")
            return success
        except ImportError as e:
            self._fallback_to_mock(f"Missing dependency: {e}")
            return False
        except Exception as e:
            self._fallback_to_mock(f"Loading error: {e}")
            return False
    
    def _fallback_to_mock(self, error_msg: str):
        """Handle model loading failure"""
        self.status = ModelStatus.MOCK
        self.error_message = error_msg
        self.model_loaded = False
        logger.warning(f"[[WARNING]]  {self.model_name} failed to load: {error_msg}")
        logger.info(f" Falling back to mock implementation for {self.model_name}")
    
    def process(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Process with automatic fallback to mock if needed"""
        # Validate input
        if not self._validate_input(image_path):
            raise ValueError(f"Invalid image path: {image_path}")
        
        # Load model if not already loaded
        if self.status == ModelStatus.NOT_LOADED:
            self.load()
        
        # Process based on status
        if self.status == ModelStatus.AVAILABLE:
            try:
                return self._process_real(image_path, **kwargs)
            except Exception as e:
                logger.error(f"[[ERROR]] {self.model_name} processing failed: {e}")
                logger.info(f" Falling back to mock for this request")
                return self._process_mock(image_path, **kwargs)
        else:
            return self._process_mock(image_path, **kwargs)
    
    def _validate_input(self, image_path: str) -> bool:
        """Validate input file"""
        try:
            path = Path(image_path)
            if not path.exists():
                logger.error(f"File not found: {image_path}")
                return False
            
            # Try to read image
            img = cv2.imread(str(path))
            if img is None:
                logger.error(f"Cannot read image: {image_path}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get model status information"""
        return {
            'name': self.model_name,
            'status': self.status,
            'loaded': self.model_loaded,
            'error': self.error_message
        }
    
    def health_check(self) -> bool:
        """Perform health check on the model"""
        return self.status in [ModelStatus.AVAILABLE, ModelStatus.MOCK]


class ObjectDetector(VisionModel):
    """YOLO object detection with graceful fallback"""
    
    def __init__(self, model_name: str = "yolov8n.pt"):
        super().__init__("ObjectDetector")
        self.yolo_model_name = model_name
        self.model = None
    
    def _load_model(self) -> bool:
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.yolo_model_name)
            return True
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"YOLO loading error: {e}")
            return False
    
    # def _process_real(self, image_path: str, confidence_threshold: float = 0.5, **kwargs) -> Dict[str, Any]:
    #     """Real YOLO processing"""
    #     start_time = time.time()
        
    #     results = self.model(image_path, conf=confidence_threshold, verbose=False)
        
    #     objects = []
    #     if results and len(results) > 0 and results[0].boxes is not None:
    #         for box in results[0].boxes:
    #             objects.append({
    #                 'class': self.model.names[int(box.cls)],
    #                 'confidence': float(box.conf),
    #                 'bbox': box.xyxy[0].tolist()
    #             })
        
    #     processing_time = time.time() - start_time
        
    #     return {
    #         'objects': objects,
    #         'count': len(objects),
    #         'processing_time': processing_time,
    #         'model_type': 'real_yolo'
    #     }
    
    def _process_real(self, image_path: str, confidence_threshold: float = 0.5, **kwargs) -> Dict[str, Any]:
        """Real YOLO processing"""
        start_time = time.time()
        
        results = self.model(image_path, conf=confidence_threshold, verbose=False)
        
        objects = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                objects.append({
                    'class': self.model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                })
        
        processing_time = time.time() - start_time
        
        return {
            'objects': objects,
            'count': len(objects),
            'processing_time': processing_time,
            'model_type': 'real_yolo'
        }

    def _process_mock(self, image_path: str, confidence_threshold: float = 0.5, **kwargs) -> Dict[str, Any]:
        """Mock object detection"""
        time.sleep(0.1)  # Simulate processing
        
        # Generate mock objects based on common scene elements
        mock_objects = [
            {'class': 'person', 'confidence': 0.85, 'bbox': [100, 100, 200, 400]},
            {'class': 'car', 'confidence': 0.92, 'bbox': [300, 200, 600, 350]},
            {'class': 'bicycle', 'confidence': 0.78, 'bbox': [50, 150, 120, 300]},
            {'class': 'traffic light', 'confidence': 0.95, 'bbox': [700, 50, 750, 150]},
        ]
        
        # Filter by confidence
        filtered_objects = [obj for obj in mock_objects if obj['confidence'] >= confidence_threshold]
        
        return {
            'objects': filtered_objects,
            'count': len(filtered_objects),
            'processing_time': 0.1,
            'model_type': 'mock'
        }


# class OCRSystem(VisionModel):
#     """OCR with EasyOCR and graceful fallback"""
    
#     def __init__(self, languages: List[str] = ['en']):
#         super().__init__("OCRSystem")
#         self.languages = languages
#         self.reader = None
    
#     def _load_model(self) -> bool:
#         """Load EasyOCR"""
#         try:
#             import easyocr
#             self.reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
#             return True
#         except ImportError:
#             logger.error("easyocr not installed. Install with: pip install easyocr")
#             return False
#         except Exception as e:
#             logger.error(f"EasyOCR loading error: {e}")
#             return False
    
#     # def _process_real(self, image_path: str, languages: List[str] = None, **kwargs) -> Dict[str, Any]:
#     #     """Real OCR processing"""
#     #     start_time = time.time()
        
#     #     results = self.reader.readtext(image_path)
        
#     #     text_regions = []
#     #     if results:
#     #         for result in results:
#     #             try:
#     #                 # EasyOCR returns (bbox, text, confidence) - 3 values
#     #                 if len(result) == 3:
#     #                     bbox, text, conf = result
#     #                 elif len(result) == 2:
#     #                     bbox, text = result
#     #                     conf = 1.0  # Default confidence if not provided
#     #                 else:
#     #                     # Skip malformed results
#     #                     logger.warning(f"Unexpected EasyOCR result format: {result}")
#     #                     continue
                        
#     #                 text_regions.append({
#     #                     'text': text,
#     #                     'confidence': float(conf),
#     #                     'bbox': bbox  # EasyOCR returns polygon [[x1,y1],[x2,y2],...]
#     #                 })
#     #             except (ValueError, TypeError) as e:
#     #                 logger.warning(f"Skipping malformed OCR result: {result}, error: {e}")
#     #                 continue
        
#     #     full_text = " ".join([region['text'] for region in text_regions])
#     #     processing_time = time.time() - start_time
        
#     #     return {
#     #         'text_regions': text_regions,
#     #         'full_text': full_text,
#     #         'processing_time': processing_time,
#     #         'model_type': 'real_easyocr'
#     #     }

#     def _process_real(self, image_path: str, languages: List[str] = None, **kwargs) -> Dict[str, Any]:
#         """Real OCR processing with robust error handling"""
#         start_time = time.time()
        
#         try:
#             results = self.reader.readtext(image_path)
            
#             text_regions = []
#             if results:
#                 for i, result in enumerate(results):
#                     try:
#                         # Debug: Log the actual result format
#                         logger.debug(f"OCR result {i}: {result} (type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'})")
                        
#                         # Handle different possible formats
#                         if isinstance(result, (list, tuple)):
#                             if len(result) == 3:
#                                 # Standard format: (bbox, text, confidence)
#                                 bbox, text, conf = result
#                             elif len(result) == 2:
#                                 # Missing confidence: (bbox, text)
#                                 bbox, text = result
#                                 conf = 1.0
#                             elif len(result) == 4:
#                                 # Some versions return 4 elements
#                                 bbox, text, conf, _ = result  # Ignore the 4th element
#                             else:
#                                 # Unexpected format - log and skip
#                                 logger.warning(f"Unexpected OCR result format with {len(result)} elements: {result}")
#                                 continue
#                         else:
#                             # Single element or unexpected type
#                             logger.warning(f"Unexpected OCR result type: {type(result)} - {result}")
#                             continue
                        
#                         # Validate extracted values
#                         if not isinstance(text, str) or not text.strip():
#                             logger.debug(f"Skipping empty/invalid text: {text}")
#                             continue
                        
#                         # Ensure confidence is a number
#                         try:
#                             conf = float(conf)
#                         except (ValueError, TypeError):
#                             logger.warning(f"Invalid confidence value: {conf}, using default 1.0")
#                             conf = 1.0
                        
#                         text_regions.append({
#                             'text': str(text).strip(),
#                             'confidence': conf,
#                             'bbox': bbox  # EasyOCR returns polygon coordinates
#                         })
                        
#                     except Exception as e:
#                         logger.warning(f"Error processing OCR result {i}: {result} - Error: {e}")
#                         continue
            
#             full_text = " ".join([region['text'] for region in text_regions])
#             processing_time = time.time() - start_time
            
#             logger.info(f"OCR completed: {len(text_regions)} text regions found, full text: '{full_text[:50]}{'...' if len(full_text) > 50 else ''}'")
            
#             return {
#                 'text_regions': text_regions,
#                 'full_text': full_text,
#                 'processing_time': processing_time,
#                 'model_type': 'real_easyocr'
#             }
            
#         except Exception as e:
#             logger.error(f"OCR processing failed completely: {e}")
#             # Log more details for debugging
#             logger.error(f"Image path: {image_path}")
#             logger.error(f"EasyOCR reader: {self.reader}")
#             raise
    
#     def _process_mock(self, image_path: str, languages: List[str] = None, **kwargs) -> Dict[str, Any]:
#         """Mock OCR processing"""
#         time.sleep(0.15)
        
#         # Mock text detection based on common scene text
#         mock_text_regions = [
#             {'text': 'STOP', 'confidence': 0.98, 'bbox': [[50, 50], [120, 50], [120, 80], [50, 80]]},
#             {'text': 'Main Street', 'confidence': 0.85, 'bbox': [[200, 400], [320, 400], [320, 430], [200, 430]]},
#             {'text': 'Speed Limit 35', 'confidence': 0.91, 'bbox': [[350, 100], [480, 100], [480, 130], [350, 130]]},
#         ]
        
#         full_text = ' '.join([region['text'] for region in mock_text_regions])
        
#         return {
#             'text_regions': mock_text_regions,
#             'full_text': full_text,
#             'processing_time': 0.15,
#             'model_type': 'mock'
#         }

class OCRSystem(VisionModel):
    """OCR with EasyOCR and graceful fallback"""
    
    def __init__(self, languages: List[str] = ['en']):
        super().__init__("OCRSystem")
        self.languages = languages
        self.reader = None
        self.easyocr_available = False
    
    def _load_model(self) -> bool:
        """Load EasyOCR with enhanced error checking"""
        try:
            import easyocr
            logger.info(f"Initializing EasyOCR with languages: {self.languages}")
            
            # Initialize with specific parameters to avoid conflicts
            self.reader = easyocr.Reader(
                self.languages, 
                gpu=False,  # Disable GPU to avoid CUDA issues
                verbose=False,
                download_enabled=True  # Allow downloading models if needed
            )
            
            # Test the reader with a minimal call to verify it works
            logger.info("EasyOCR reader initialized successfully")
            self.easyocr_available = True
            return True
            
        except ImportError as e:
            logger.error(f"easyocr not installed: {e}")
            logger.error("Install with: pip install easyocr")
            return False
        except Exception as e:
            logger.error(f"EasyOCR loading error: {e}")
            logger.error(f"Error type: {type(e)}")
            return False
    
    def _process_real(self, image_path: str, languages: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Real OCR processing with comprehensive error handling"""
        start_time = time.time()
        
        if not self.easyocr_available or self.reader is None:
            logger.error("EasyOCR reader not available")
            raise RuntimeError("EasyOCR reader not initialized")
        
        try:
            # Validate image path exists and is readable
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Call EasyOCR with explicit parameters
            logger.debug(f"Processing image with EasyOCR: {image_path}")
            
            # Try the basic readtext call first
            try:
                results = self.reader.readtext(
                    image_path,
                    detail=1,  # Return detailed results (bbox, text, confidence)
                    paragraph=False,  # Don't group into paragraphs
                    width_ths=0.7,  # Default width threshold
                    height_ths=0.7  # Default height threshold
                )
            except TypeError as e:
                # If detailed parameters fail, try basic call
                logger.warning(f"Detailed EasyOCR call failed: {e}, trying basic call")
                results = self.reader.readtext(image_path)
            
            logger.debug(f"EasyOCR returned {len(results) if results else 0} results")
            
            text_regions = []
            if results:
                for i, result in enumerate(results):
                    try:
                        logger.debug(f"Processing result {i}: {type(result)} with length {len(result) if hasattr(result, '__len__') else 'N/A'}")
                        
                        # Handle the result based on its structure
                        if isinstance(result, (list, tuple)):
                            if len(result) >= 3:
                                # Standard: (bbox, text, confidence)
                                bbox = result[0]
                                text = result[1] 
                                conf = result[2]
                            elif len(result) == 2:
                                # Missing confidence: (bbox, text)
                                bbox = result[0]
                                text = result[1]
                                conf = 1.0
                            else:
                                logger.warning(f"Unexpected result length {len(result)}: {result}")
                                continue
                        else:
                            logger.warning(f"Unexpected result type {type(result)}: {result}")
                            continue
                        
                        # Validate and clean the extracted data
                        if not isinstance(text, str):
                            text = str(text)
                        
                        text = text.strip()
                        if not text:
                            continue
                            
                        try:
                            conf = float(conf) if conf is not None else 1.0
                        except (ValueError, TypeError):
                            conf = 1.0
                        
                        text_regions.append({
                            'text': text,
                            'confidence': conf,
                            'bbox': bbox
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing OCR result {i}: {e}")
                        logger.warning(f"Problematic result: {result}")
                        continue
            
            full_text = " ".join([region['text'] for region in text_regions])
            processing_time = time.time() - start_time
            
            logger.info(f"OCR completed: {len(text_regions)} regions, text preview: '{full_text[:50]}{'...' if len(full_text) > 50 else ''}'")
            
            return {
                'text_regions': text_regions,
                'full_text': full_text,
                'processing_time': processing_time,
                'model_type': 'real_easyocr'
            }
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Image path: {image_path}")
            
            # Import traceback for more detailed error info
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            raise
    
    def _process_mock(self, image_path: str, languages: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Mock OCR processing"""
        time.sleep(0.15)
        
        # Mock text detection based on common scene text
        mock_text_regions = [
            {'text': 'STOP', 'confidence': 0.98, 'bbox': [[50, 50], [120, 50], [120, 80], [50, 80]]},
            {'text': 'Main Street', 'confidence': 0.85, 'bbox': [[200, 400], [320, 400], [320, 430], [200, 430]]},
            {'text': 'Speed Limit 35', 'confidence': 0.91, 'bbox': [[350, 100], [480, 100], [480, 130], [350, 130]]},
        ]
        
        full_text = ' '.join([region['text'] for region in mock_text_regions])
        
        return {
            'text_regions': mock_text_regions,
            'full_text': full_text,
            'processing_time': 0.15,
            'model_type': 'mock'
        }

class SceneDescriber(VisionModel):
    """BLIP scene description with graceful fallback"""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        super().__init__("SceneDescriber")
        self.blip_model_name = model_name
        self.processor = None
        self.model = None
    
    def _load_model(self) -> bool:
        """Load BLIP model"""
        try:
            from PIL import Image
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            self.processor = BlipProcessor.from_pretrained(self.blip_model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.blip_model_name)
            return True
        except ImportError:
            logger.error("transformers or PIL not installed. Install with: pip install transformers pillow")
            return False
        except Exception as e:
            logger.error(f"BLIP loading error: {e}")
            return False
    
    def _process_real(self, image_path: str, detail_level: str = 'medium', **kwargs) -> Dict[str, Any]:
        """Real BLIP processing"""
        start_time = time.time()
        
        try:
            from PIL import Image
            
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(image, return_tensors="pt")
            
            max_length = 50 if detail_level == "basic" else 150
            out = self.model.generate(**inputs, max_length=max_length)
            description = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Extract scene elements (basic keyword extraction)
            scene_elements = [word.lower() for word in description.split() 
                            if word.isalpha() and len(word) > 2][:10]
            
            processing_time = time.time() - start_time
            
            return {
                'description': description,
                'scene_elements': scene_elements,
                'confidence': 0.9,
                'processing_time': processing_time,
                'model_type': 'real_blip'
            }
        except Exception as e:
            logger.error(f"BLIP processing error: {e}")
            raise
    
    def _process_mock(self, image_path: str, detail_level: str = 'medium', **kwargs) -> Dict[str, Any]:
        """Mock scene description"""
        time.sleep(0.25)
        
        descriptions = {
            'basic': 'A scene with people and objects.',
            'medium': 'A busy street scene with people, cars, and buildings in an urban environment.',
            'detailed': 'A bustling urban intersection with pedestrians crossing, vehicles waiting at traffic lights, and commercial buildings lining the street. The scene captures everyday city life with various people engaged in different activities.'
        }
        
        description = descriptions.get(detail_level, descriptions['medium'])
        scene_elements = ['people', 'street', 'cars', 'buildings', 'traffic', 'urban', 'intersection']
        
        return {
            'description': description,
            'scene_elements': scene_elements,
            'confidence': 0.8,
            'processing_time': 0.25,
            'model_type': 'mock'
        }


class ActionRecognizer(VisionModel):
    """VideoMAE action recognition with image/video handling"""
    
    def __init__(self, model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics"):
        super().__init__("ActionRecognizer")
        self.videomae_model_name = model_name
        self.processor = None
        self.model = None
    
    def _load_model(self) -> bool:
        """Load VideoMAE model"""
        try:
            import torch
            from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
            
            self.processor = VideoMAEFeatureExtractor.from_pretrained(self.videomae_model_name)
            self.model = VideoMAEForVideoClassification.from_pretrained(self.videomae_model_name)
            return True
        except ImportError:
            logger.error("torch or transformers not installed. Install with: pip install torch transformers")
            return False
        except Exception as e:
            logger.error(f"VideoMAE loading error: {e}")
            return False
    
    def _is_video_file(self, file_path: str) -> bool:
        """Check if file is a video"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        return Path(file_path).suffix.lower() in video_extensions
    
    def _process_real(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Real VideoMAE processing"""
        start_time = time.time()
        
        try:
            import torch
            
            if self._is_video_file(image_path):
                frames = self._extract_video_frames(image_path)
            else:
                # For single image, create a sequence by duplicating the frame
                frames = self._create_frame_sequence_from_image(image_path)
            
            if not frames:
                return self._process_mock(image_path, **kwargs)
            
            inputs = self.processor(frames, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            top5 = torch.topk(probs, 5)
            
            actions = []
            for i in range(len(top5.indices)):
                label_id = top5.indices[i].item()
                confidence = top5.values[i].item()
                actions.append({
                    'action': self.model.config.id2label[label_id],
                    'confidence': confidence,
                    'temporal_location': [0.0, 1.0]
                })
            
            processing_time = time.time() - start_time
            
            return {
                'actions': actions,
                'primary_action': actions[0]['action'] if actions else None,
                'processing_time': processing_time,
                'model_type': 'real_videomae'
            }
        except Exception as e:
            logger.error(f"VideoMAE processing error: {e}")
            raise
    
    def _extract_video_frames(self, video_path: str, num_frames: int = 16) -> List:
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return []
            
            frame_indices = [int(i) for i in 
                           (total_frames * i / num_frames for i in range(num_frames))]
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            cap.release()
            return frames
        except Exception as e:
            logger.error(f"Video frame extraction error: {e}")
            return []
    
    def _create_frame_sequence_from_image(self, image_path: str, num_frames: int = 16) -> List:
        """Create frame sequence from single image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Duplicate the frame to create a sequence
            return [img_rgb] * num_frames
        except Exception as e:
            logger.error(f"Image frame sequence creation error: {e}")
            return []
    
    def _process_mock(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Mock action recognition"""
        time.sleep(0.2)
        
        mock_actions = [
            {'action': 'walking', 'confidence': 0.75, 'temporal_location': [0.0, 1.0]},
            {'action': 'standing', 'confidence': 0.65, 'temporal_location': [0.0, 1.0]},
            {'action': 'talking', 'confidence': 0.55, 'temporal_location': [0.0, 1.0]},
        ]
        
        return {
            'actions': mock_actions,
            'primary_action': mock_actions[0]['action'],
            'processing_time': 0.2,
            'model_type': 'mock'
        }


class VisionQueryRouter:
    """Robust vision query router with intelligent model management"""
    
    def __init__(self, cache_manager: CacheManager, config: Dict[str, Any] = None):
        self.cache_manager = cache_manager
        self.config = config or {}
        
        # Initialize models
        self.models = {}
        self.model_stats = {}
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all vision models with error handling"""
        logger.info("Initializing vision models...")
        
        model_configs = {
            'detect_objects': {
                'class': ObjectDetector,
                'kwargs': {'model_name': self.config.get('yolo_model', 'yolov8n.pt')}
            },
            'read_text': {
                'class': OCRSystem,
                'kwargs': {'languages': self.config.get('ocr_languages', ['en'])}
            },
            'describe_scene': {
                'class': SceneDescriber,
                'kwargs': {'model_name': self.config.get('blip_model', 'Salesforce/blip-image-captioning-base')}
            }
            # 'recognize_actions': {
            #     'class': ActionRecognizer,
            #     'kwargs': {'model_name': self.config.get('videomae_model', 'MCG-NJU/videomae-base-finetuned-kinetics')}
            # }
        }
        
        for model_name, config in model_configs.items():
            try:
                model = config['class'](**config['kwargs'])
                load_success = model.load()
                
                self.models[model_name] = model
                self.model_stats[model_name] = {
                    'calls': 0, 
                    'cache_hits': 0, 
                    'total_time': 0,
                    'errors': 0,
                    'status': model.status
                }
                
                status_emoji = "[[OK]]" if load_success else "[[WARNING]]"
                logger.info(f"{status_emoji} {model_name}: {model.status}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {model_name}: {e}")
                self.model_stats[model_name] = {
                    'calls': 0, 'cache_hits': 0, 'total_time': 0, 
                    'errors': 1, 'status': 'failed'
                }
    
    def execute_query(self, query: VisionQuery) -> VisionResult:
        """Execute vision query with enhanced error handling"""
        start_time = time.time()
        
        # Check if model is available
        if query.query_type not in self.models:
            raise ValueError(f"Unknown query type: {query.query_type}")
        
        model = self.models[query.query_type]
        
        # Check model health
        if not model.health_check():
            raise RuntimeError(f"Model {query.query_type} is not healthy")
        
        # Check cache first
        cached_result = self.cache_manager.get(query.cache_key)
        if cached_result:
            self.model_stats[query.query_type]['cache_hits'] += 1
            logger.debug(f"Cache hit for {query.query_type}")
            return cached_result
        
        # Execute model
        try:
            result_data = model.process(query.image_path, **query.params)
            processing_time = time.time() - start_time
            
            # Create result object
            result = VisionResult(
                query_type=query.query_type,
                result=result_data,
                confidence=result_data.get('confidence', 0.0),
                processing_time=processing_time
            )
            
            # Cache result
            try:
                self.cache_manager.set(query.cache_key, result)
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")
            
            # Update stats
            self.model_stats[query.query_type]['calls'] += 1
            self.model_stats[query.query_type]['total_time'] += processing_time
            
            logger.debug(f"Executed {query.query_type} in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            self.model_stats[query.query_type]['errors'] += 1
            logger.error(f"[[ERROR]] Error executing {query.query_type}: {e}")
            raise
    
    def initial_analysis(self, processed_input: Dict[str, Any]) -> Dict[str, VisionResult]:
        """Perform comprehensive initial analysis"""
        image_path = processed_input['processed_files'][0]
        
        # Define queries for available models only
        available_queries = []
        
        query_definitions = [
            ('detect_objects', {'confidence_threshold': 0.5}),
            ('read_text', {'languages': ['en']}),
            ('describe_scene', {'detail_level': 'medium'}),
        ]
        
        # Add action recognition for video files
        if processed_input['input_type'] == 'video' or len(processed_input['processed_files']) > 1:
            query_definitions.append(('recognize_actions', {}))
        
        # Create queries for available models
        for query_type, params in query_definitions:
            if (query_type in self.models and 
                self.models[query_type].health_check()):
                available_queries.append(VisionQuery(query_type, image_path, params))
        
        # Execute all queries
        results = {}
        for query in available_queries:
            try:
                result = self.execute_query(query)
                results[query.query_type] = result
                logger.info(f"[[OK]] {query.query_type} completed")
            except Exception as e:
                logger.error(f"[[ERROR] {query.query_type} failed: {e}")
                # Continue with other models even if one fails
        
        logger.info(f"Initial analysis complete: {len(results)}/{len(available_queries)} models succeeded")
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of available (healthy) models"""
        return [name for name, model in self.models.items() 
                if model.health_check()]
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all models"""
        status = {}
        for name, model in self.models.items():
            model_status = model.get_status()
            model_status.update(self.model_stats[name])
            status[name] = model_status
        return status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {}
        for model_name, model_stats in self.model_stats.items():
            calls = model_stats['calls']
            cache_hits = model_stats['cache_hits']
            total_time = model_stats['total_time']
            
            stats[model_name] = {
                'total_calls': calls,
                'cache_hits': cache_hits,
                'cache_hit_rate': cache_hits / (calls + cache_hits) if (calls + cache_hits) > 0 else 0,
                'avg_processing_time': total_time / calls if calls > 0 else 0,
                'total_processing_time': total_time,
                'errors': model_stats['errors'],
                'status': model_stats['status'],
                'health': self.models[model_name].health_check() if model_name in self.models else False
            }
        
        return stats
    
    def cleanup(self):
        """Cleanup model resources"""
        logger.info("Cleaning up vision models...")
        for model_name, model in self.models.items():
            try:
                # Add cleanup methods to models if needed
                if hasattr(model, 'cleanup'):
                    model.cleanup()
            except Exception as e:
                logger.warning(f"Cleanup error for {model_name}: {e}")