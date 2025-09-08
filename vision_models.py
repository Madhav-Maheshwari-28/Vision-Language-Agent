"""
Vision models and query routing system for the Vision-Language Agent.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any

from data_structures import VisionQuery, VisionResult
from input_processing import CacheManager

logger = logging.getLogger(__name__)


class VisionModel(ABC):
    """Abstract base class for vision models"""
    
    @abstractmethod
    def process(self, image_path: str, **kwargs) -> Dict[str, Any]:
        pass


# class ObjectDetector(VisionModel):
    """Mock object detection model (replace with actual YOLO/DETR)"""
    
    def __init__(self):
        self.model_loaded = True  # Mock initialization
        
    def process(self, image_path: str, confidence_threshold: float = 0.5, **kwargs) -> Dict[str, Any]:
        """Detect objects in image"""
        time.sleep(0.1)  # Simulate processing time
        
        # Mock detection results
        objects = [
            {'class': 'person', 'confidence': 0.95, 'bbox': [100, 100, 200, 300]},
            {'class': 'car', 'confidence': 0.87, 'bbox': [300, 200, 500, 350]},
            {'class': 'traffic_light', 'confidence': 0.92, 'bbox': [50, 50, 80, 120]}
        ]
        
        # Filter by confidence
        filtered_objects = [obj for obj in objects if obj['confidence'] >= confidence_threshold]
        
        return {
            'objects': filtered_objects,
            'count': len(filtered_objects),
            'processing_time': 0.1
        }

from ultralytics import YOLO

class ObjectDetector(VisionModel):
    """YOLO object detection model (replaces mock)"""

    def __init__(self, model_name="yolov8n.pt"):
        # Load YOLO model (you can switch to yolov8s.pt, yolov8m.pt, etc.)
        self.model = YOLO(model_name)
        self.model_loaded = True

    def process(self, image_path: str, confidence_threshold: float = 0.5, **kwargs) -> Dict[str, Any]:
        """Detect objects in image using YOLO"""
        start_time = time.time()

        # Run YOLO inference
        results = self.model(image_path, conf=confidence_threshold)

        # Collect detections
        objects = []
        for r in results[0].boxes:
            objects.append({
                'class': self.model.names[int(r.cls)],          # label
                'confidence': float(r.conf),                    # score
                'bbox': r.xyxy[0].tolist()                      # [x1, y1, x2, y2]
            })

        processing_time = time.time() - start_time

        return {
            'objects': objects,
            'count': len(objects),
            'processing_time': processing_time
        }


# class OCRSystem(VisionModel):
    """Mock OCR system (replace with EasyOCR/PaddleOCR)"""
    
    def __init__(self):
        self.model_loaded = True
        
    def process(self, image_path: str, languages: List[str] = ['en'], **kwargs) -> Dict[str, Any]:
        """Extract text from image"""
        time.sleep(0.15)  # Simulate processing time
        
        # Mock OCR results
        text_regions = [
            {'text': 'DON\'T WALK', 'confidence': 0.98, 'bbox': [50, 50, 120, 80]},
            {'text': 'MAIN ST', 'confidence': 0.85, 'bbox': [200, 400, 280, 420]},
            {'text': 'SPEED LIMIT 35', 'confidence': 0.91, 'bbox': [350, 100, 450, 130]}
        ]
        
        full_text = ' '.join([region['text'] for region in text_regions])
        
        return {
            'text_regions': text_regions,
            'full_text': full_text,
            'processing_time': 0.15
        }

from paddleocr import PaddleOCR

class OCRSystem(VisionModel):
    """OCR system using PaddleOCR (replaces mock)"""

    def __init__(self, languages: List[str] = ['en']):
        # Initialize PaddleOCR with given languages
        lang_str = ",".join(languages) if isinstance(languages, list) else languages
        self.reader = PaddleOCR(lang=lang_str, use_angle_cls=True)
        self.model_loaded = True

    def process(self, image_path: str, languages: List[str] = ['en'], **kwargs) -> Dict[str, Any]:
        """Extract text from image using PaddleOCR"""
        start_time = time.time()

        # Run OCR
        results = self.reader.ocr(image_path, cls=True)

        text_regions = []
        for line in results[0]:  # results[0] contains list of detected text
            bbox, (text, conf) = line
            text_regions.append({
                'text': text,
                'confidence': float(conf),
                'bbox': [list(map(int, point)) for point in bbox]  # polygon bbox
            })

        full_text = " ".join([region['text'] for region in text_regions])
        processing_time = time.time() - start_time

        return {
            'text_regions': text_regions,
            'full_text': full_text,
            'processing_time': processing_time
        }

# class ActionRecognizer(VisionModel):
    """Mock action recognition (replace with SlowFast/VideoMAE)"""
    
    def __init__(self):
        self.model_loaded = True
        
    def process(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Recognize actions in image/video"""
        time.sleep(0.2)  # Simulate processing time
        
        # Mock action recognition results
        actions = [
            {'action': 'running', 'confidence': 0.89, 'temporal_location': [0.2, 0.8]},
            {'action': 'crossing_street', 'confidence': 0.93, 'temporal_location': [0.1, 0.9]}
        ]
        
        return {
            'actions': actions,
            'primary_action': actions[0]['action'] if actions else None,
            'processing_time': 0.2
        }

import torch
import cv2
from typing import Dict, Any
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification

class ActionRecognizer(VisionModel):
    """VideoMAE action recognition (replaces mock)"""
    
    def __init__(self, model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics"):
        self.processor = VideoMAEFeatureExtractor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        self.model_loaded = True

    def process(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Recognize actions in video using VideoMAE"""
        start_time = time.time()
        
        # Read video frames
        cap = cv2.VideoCapture(image_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = max(total_frames // 16, 1)  # sample ~16 frames

        for i in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        if not frames:
            return {
                'actions': [],
                'primary_action': None,
                'processing_time': time.time() - start_time
            }

        # Preprocess and run through model
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
                'temporal_location': [0.0, 1.0]  # full clip since VideoMAE is clip-level
            })

        processing_time = time.time() - start_time

        return {
            'actions': actions,
            'primary_action': actions[0]['action'] if actions else None,
            'processing_time': processing_time
        }


class SceneDescriber(VisionModel):
    """Mock scene description (replace with BLIP-2/LLaVA)"""
    
    def __init__(self):
        self.model_loaded = True
        
    def process(self, image_path: str, detail_level: str = 'medium', **kwargs) -> Dict[str, Any]:
        """Generate natural language description of scene"""
        time.sleep(0.25)  # Simulate processing time
        
        # Mock scene descriptions based on detail level
        descriptions = {
            'basic': 'A person crossing a street.',
            'medium': 'A person in casual clothing is running across a busy street intersection with traffic lights and cars.',
            'detailed': 'A person wearing a red jacket and dark pants is running across a four-way intersection. There are several cars waiting at traffic lights, and the pedestrian signal shows "DON\'T WALK". The scene appears to be in an urban setting with commercial buildings in the background.'
        }
        
        description = descriptions.get(detail_level, descriptions['medium'])
        
        return {
            'description': description,
            'scene_elements': ['person', 'street', 'cars', 'traffic_lights', 'buildings'],
            'confidence': 0.88,
            'processing_time': 0.25
        }

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class SceneDescriber(VisionModel):
    """BLIP image captioning (replaces mock scene description)"""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model_loaded = True
        
    def process(self, image_path: str, detail_level: str = 'medium', **kwargs) -> Dict[str, Any]:
        """Generate natural language description of scene using BLIP"""
        start_time = time.time()
        
        # Open image
        image = Image.open(image_path).convert("RGB")
        
        # Generate caption
        inputs = self.processor(image, return_tensors="pt")
        out = self.model.generate(**inputs, 
                                  max_length=50 if detail_level == "basic" else 150)
        description = self.processor.decode(out[0], skip_special_tokens=True)
        
        # For consistency, extract scene elements (very basic keyword extraction)
        scene_elements = [word for word in description.split() if word.isalpha()]
        
        processing_time = time.time() - start_time
        
        return {
            'description': description,
            'scene_elements': scene_elements[:10],  # limit to 10 elements
            'confidence': 0.9,  # BLIP doesn’t output confidence, so fixed score
            'processing_time': processing_time
        }


class VisionQueryRouter:
    """Routes vision queries to appropriate models with caching"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        
        # Initialize vision models (excluding ActionRecognizer for now)
        self.models = {
            'detect_objects': ObjectDetector(),
            'read_text': OCRSystem(),
            'describe_scene': SceneDescriber()
        }
        
        # Model execution statistics
        self.model_stats = {model: {'calls': 0, 'cache_hits': 0, 'total_time': 0} 
                           for model in self.models.keys()}
    
    def execute_query(self, query: VisionQuery) -> VisionResult:
        """Execute vision query with caching"""
        start_time = time.time()
        
        # Check cache first
        cached_result = self.cache_manager.get(query.cache_key)
        if cached_result:
            self.model_stats[query.query_type]['cache_hits'] += 1
            logger.info(f"Cache hit for {query.query_type}")
            return cached_result
        
        # Execute model
        if query.query_type not in self.models:
            raise ValueError(f"Unknown query type: {query.query_type}")
        
        model = self.models[query.query_type]
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
        self.cache_manager.set(query.cache_key, result)
        
        # Update stats
        self.model_stats[query.query_type]['calls'] += 1
        self.model_stats[query.query_type]['total_time'] += processing_time
        
        logger.info(f"Executed {query.query_type} in {processing_time:.3f}s")
        return result
    
    def initial_analysis(self, processed_input: Dict[str, Any]) -> Dict[str, VisionResult]:
        """Perform initial comprehensive analysis of image (3 models)"""
        image_path = processed_input['processed_files'][0]  # Use first image/frame
        
        # Define initial queries (removed ActionRecognizer)
        initial_queries = [
            VisionQuery('detect_objects', image_path, {'confidence_threshold': 0.5}),
            VisionQuery('read_text', image_path, {'languages': ['en']}),
            VisionQuery('describe_scene', image_path, {'detail_level': 'medium'})
        ]
        
        # Execute all initial queries
        results = {}
        for query in initial_queries:
            try:
                result = self.execute_query(query)
                results[query.query_type] = result
            except Exception as e:
                logger.error(f"Error in initial analysis {query.query_type}: {e}")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vision router statistics"""
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
                'total_processing_time': total_time
            }
        
        return stats