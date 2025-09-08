"""
Main Vision-Language Agent that orchestrates all components.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

from data_structures import ReasoningStep
from input_processing import InputProcessor, CacheManager
from vision_models import VisionQueryRouter
from memory_reasoning import MemoryManager, LLMReasoningEngine

logger = logging.getLogger(__name__)


class VisionLanguageAgent:
    """Main agent that orchestrates all components"""
    
    def __init__(self):
        # Initialize components
        self.input_processor = InputProcessor()
        self.cache_manager = CacheManager()
        self.vision_router = VisionQueryRouter(self.cache_manager)
        self.memory_manager = MemoryManager()
        self.reasoning_engine = LLMReasoningEngine(self.vision_router)
        
        # System statistics
        self.system_stats = {
            'total_queries': 0,
            'successful_responses': 0,
            'error_count': 0,
            'average_response_time': 0.0
        }
    
    async def process_query(self, image_path: str, question: str, session_id: str = None) -> Dict[str, Any]:
        """Main query processing function"""
        start_time = time.time()
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        try:
            # Step 1: Input processing
            logger.info("Processing input...")
            processed_input = self.input_processor.process_input(image_path, question)
            
            # Step 2: Memory retrieval
            logger.info("Retrieving conversation context...")
            context = self.memory_manager.get_context(session_id)
            
            # Step 3: Initial vision analysis
            logger.info("Performing initial vision analysis...")
            initial_features = self.vision_router.initial_analysis(processed_input)
            
            # Step 4: Multi-step reasoning
            logger.info("Starting reasoning process...")
            response, reasoning_chain = self.reasoning_engine.reason(
                question, context, initial_features
            )
            
            processing_time = time.time() - start_time
            
            # Step 5: Memory update
            logger.info("Updating conversation memory...")
            self.memory_manager.update(
                session_id, question, response, image_path, reasoning_chain, processing_time
            )
            
            # Update system statistics
            self.system_stats['total_queries'] += 1
            self.system_stats['successful_responses'] += 1
            self.system_stats['average_response_time'] = (
                (self.system_stats['average_response_time'] * (self.system_stats['total_queries'] - 1) + processing_time) 
                / self.system_stats['total_queries']
            )
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            
            return {
                'session_id': session_id,
                'response': response,
                'reasoning_chain': [asdict(step) for step in reasoning_chain],
                'processing_time': processing_time,
                'visual_features': {k: asdict(v) for k, v in initial_features.items()},
                'success': True
            }
            
        except Exception as e:
            self.system_stats['total_queries'] += 1
            self.system_stats['error_count'] += 1
            processing_time = time.time() - start_time
            
            logger.error(f"Error processing query: {e}")
            
            return {
                'session_id': session_id,
                'response': f"I encountered an error processing your query: {str(e)}",
                'error': str(e),
                'processing_time': processing_time,
                'success': False
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'system': self.system_stats,
            'cache': self.cache_manager.get_stats(),
            'vision_router': self.vision_router.get_stats()
        }
    
    def cleanup(self):
        """Cleanup system resources"""
        # Clean up temporary files
        temp_dir = Path("temp_processing")
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
        
        logger.info("System cleanup completed")