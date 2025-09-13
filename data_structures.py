"""
Core data structures for the Vision-Language Agent system.
"""
# At the top of your files:
import sys
sys.stdout.reconfigure(encoding='utf-8')
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class VisionQuery:
    """Represents a vision query with parameters"""
    query_type: str
    image_path: str
    params: Dict[str, Any]
    timestamp: datetime = None
    cache_key: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.cache_key is None:
            self.cache_key = self.generate_cache_key()
    
    def generate_cache_key(self) -> str:
        """Generate unique cache key for this query"""
        key_data = f"{self.query_type}_{self.image_path}_{json.dumps(self.params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()


@dataclass
class VisionResult:
    """Result from vision model execution"""
    query_type: str
    result: Dict[str, Any]
    confidence: float
    processing_time: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ReasoningStep:
    """Individual step in reasoning chain"""
    step_number: int
    reasoning: str
    evidence: List[str]
    vision_queries: List[VisionQuery]
    confidence: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ConversationTurn:
    """Single conversation exchange"""
    session_id: str
    turn_id: str
    user_input: str
    image_path: Optional[str]
    response: str
    reasoning_chain: List[ReasoningStep]
    processing_time: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.turn_id is None:
            self.turn_id = str(uuid.uuid4())