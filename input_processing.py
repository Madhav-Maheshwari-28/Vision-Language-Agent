"""
Input processing and multi-level caching system for the Vision-Language Agent.
"""
# At the top of your files:
import sys
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import logging
import numpy as np
import pickle
import redis
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class InputProcessor:
    """Handles image/video input processing and validation"""
    
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    MAX_IMAGE_SIZE = (2048, 2048)
    
    def __init__(self):
        self.temp_dir = Path("temp_processing")
        self.temp_dir.mkdir(exist_ok=True)
    
    def validate_input(self, file_path: str) -> Tuple[bool, str, str]:
        """Validate input file and determine type"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "error", "File does not exist"
        
        suffix = path.suffix.lower()
        
        if suffix in self.SUPPORTED_IMAGE_FORMATS:
            return True, "image", "Valid image file"
        elif suffix in self.SUPPORTED_VIDEO_FORMATS:
            return True, "video", "Valid video file"
        else:
            return False, "error", f"Unsupported format: {suffix}"
    
    def process_image(self, image_path: str) -> str:
        """Process and normalize image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Resize if too large
            height, width = image.shape[:2]
            if width > self.MAX_IMAGE_SIZE[0] or height > self.MAX_IMAGE_SIZE[1]:
                scale = min(self.MAX_IMAGE_SIZE[0]/width, self.MAX_IMAGE_SIZE[1]/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Save processed image
            processed_path = self.temp_dir / f"processed_{int(time.time())}_{Path(image_path).name}"
            cv2.imwrite(str(processed_path), image)
            
            return str(processed_path)
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise
    
    def extract_video_frames(self, video_path: str, num_frames: int = 8) -> List[str]:
        """Extract representative frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                raise ValueError("Could not read video frames")
            
            # Calculate frame indices to extract
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            
            extracted_frames = []
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Save frame
                    frame_path = self.temp_dir / f"frame_{int(time.time())}_{i}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(str(frame_path))
            
            cap.release()
            return extracted_frames
            
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            raise
    
    def process_input(self, file_path: str, question: str) -> Dict[str, Any]:
        """Main input processing function"""
        is_valid, input_type, message = self.validate_input(file_path)
        
        if not is_valid:
            raise ValueError(message)
        
        result = {
            'input_type': input_type,
            'question': question,
            'original_path': file_path,
            'processed_files': []
        }
        
        if input_type == 'image':
            processed_path = self.process_image(file_path)
            result['processed_files'] = [processed_path]
        elif input_type == 'video':
            frame_paths = self.extract_video_frames(file_path)
            result['processed_files'] = frame_paths
        
        return result


class CacheManager:
    """Multi-level caching system for vision results"""
    
    def __init__(self, redis_host='localhost', redis_port=6379, max_memory_cache=1000):
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
            self.redis_available = True
        except:
            logger.warning("Redis not available, using memory cache only")
            self.redis_available = False
        
        # L1 Cache - Memory (fastest)
        self.memory_cache = {}
        self.max_memory_cache = max_memory_cache
        self.cache_access_times = {}
        
        # L2 Cache - Disk persistence
        self.disk_cache_dir = Path("cache")
        self.disk_cache_dir.mkdir(exist_ok=True)
        
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'redis_hits': 0,
            'disk_hits': 0
        }
    
    def _evict_memory_cache(self):
        """Evict least recently used items from memory cache"""
        if len(self.memory_cache) >= self.max_memory_cache:
            # Remove oldest 20% of entries
            items_to_remove = int(self.max_memory_cache * 0.2)
            sorted_items = sorted(self.cache_access_times.items(), key=lambda x: x[1])
            
            for key, _ in sorted_items[:items_to_remove]:
                self.memory_cache.pop(key, None)
                self.cache_access_times.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache (L1 -> L2 -> L3)"""
        # L1 - Memory cache
        if key in self.memory_cache:
            self.cache_access_times[key] = time.time()
            self.cache_stats['hits'] += 1
            self.cache_stats['memory_hits'] += 1
            return self.memory_cache[key]
        
        # L2 - Redis cache
        if self.redis_available:
            try:
                data = self.redis_client.get(key)
                if data:
                    result = pickle.loads(data)
                    # Promote to L1
                    self._evict_memory_cache()
                    self.memory_cache[key] = result
                    self.cache_access_times[key] = time.time()
                    self.cache_stats['hits'] += 1
                    self.cache_stats['redis_hits'] += 1
                    return result
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        # L3 - Disk cache
        disk_path = self.disk_cache_dir / f"{key}.pkl"
        if disk_path.exists():
            try:
                with open(disk_path, 'rb') as f:
                    result = pickle.load(f)
                
                # Promote to L1 and L2
                self._evict_memory_cache()
                self.memory_cache[key] = result
                self.cache_access_times[key] = time.time()
                
                if self.redis_available:
                    try:
                        self.redis_client.setex(key, 3600, pickle.dumps(result))  # 1 hour TTL
                    except Exception as e:
                        logger.warning(f"Redis set error: {e}")
                
                self.cache_stats['hits'] += 1
                self.cache_stats['disk_hits'] += 1
                return result
                
            except Exception as e:
                logger.warning(f"Disk cache read error: {e}")
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set item in all cache levels"""
        # L1 - Memory
        self._evict_memory_cache()
        self.memory_cache[key] = value
        self.cache_access_times[key] = time.time()
        
        # L2 - Redis
        if self.redis_available:
            try:
                self.redis_client.setex(key, ttl, pickle.dumps(value))
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
        
        # L3 - Disk
        try:
            disk_path = self.disk_cache_dir / f"{key}.pkl"
            with open(disk_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Disk cache write error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'total_requests': total_requests
        }