"""
Main interface for the Vision-Language Agent system.
Properly integrated with the existing main_agent.py VisionLanguageAgent class.
"""
# At the top of your files:
import sys
sys.stdout.reconfigure(encoding='utf-8')
import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Import your existing modules
try:
    from main_agent import VisionLanguageAgent
except ImportError as e:
    print(f"Error importing main_agent: {e}")
    print("Make sure all your project files are in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vision_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class VisionAgentInterface:
    """Main interface for interacting with the existing Vision-Language Agent"""
    
    def __init__(self):
        self.agent = None
        self.current_session_id = None
        self.setup_environment()
    
    def setup_environment(self):
        """Setup environment and validate requirements"""
        print("Initializing Vision-Language Agent...")
        
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("Warning: OPENAI_API_KEY not found in environment variables")
            print("The reasoning engine requires an OpenAI API key to function properly")
            
            # Option to input API key
            api_key = input("Enter your OpenAI API key (or press Enter to continue without): ").strip()
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
                print("OpenAI API key set successfully")
            else:
                print("Continuing without OpenAI API key - reasoning engine may not work")
        
        # Create necessary directories
        for directory in ['temp_processing', 'cache']:
            Path(directory).mkdir(exist_ok=True)
            print(f"Created directory: {directory}")
        
        # Initialize the agent
        try:
            self.agent = VisionLanguageAgent()
            print("Vision-Language Agent initialized successfully!")
            self.display_system_status()
        except Exception as e:
            print(f"Error initializing agent: {e}")
            logger.error(f"Agent initialization failed: {e}")
            sys.exit(1)
    
    def display_system_status(self):
        """Display system status and capabilities"""
        try:
            stats = self.agent.get_system_stats()
            
            print("\n" + "="*60)
            print("VISION-LANGUAGE AGENT STATUS")
            print("="*60)
            
            # Vision router status
            vision_stats = stats.get('vision_router', {})
            print("\nVision Models Status:")
            for model_name, model_stats in vision_stats.items():
                status_symbol = "✓" if model_stats.get('health', False) else "✗"
                print(f"  {status_symbol} {model_name}: {model_stats.get('status', 'unknown')}")
            
            # Cache status
            cache_stats = stats.get('cache', {})
            print(f"\nCache System:")
            print(f"  Memory cache size: {cache_stats.get('memory_cache_size', 0)}")
            print(f"  Total cache requests: {cache_stats.get('total_requests', 0)}")
            print(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.1%}")
            
            # System capabilities
            print(f"\nSystem Capabilities:")
            print(f"  Object Detection: YOLO-based real-time detection")
            print(f"  Text Recognition: OCR with multi-language support")  
            print(f"  Scene Description: BLIP-based natural language generation")
            print(f"  Multi-step Reasoning: OpenAI LLM integration")
            print(f"  Conversation Memory: SQLite-based persistent storage")
            print(f"  Caching: Multi-level (Memory → Redis → Disk)")
            
            print("="*60)
            
        except Exception as e:
            print(f"Could not retrieve system status: {e}")
    
    def print_banner(self):
        """Print welcome banner"""
        banner = """
════════════════════════════════════════════════════════════════
                 VISION-LANGUAGE AGENT v1.0                     
                                                                
  Advanced AI system for complex visual scene understanding     
  • Multi-step reasoning with LLM integration                  
  • Real vision models (YOLO, OCR, BLIP)                       
  • Multi-turn conversational memory                           
  • Intelligent caching system                                 
════════════════════════════════════════════════════════════════
        """
        print(banner)
    
    async def process_single_query(self, image_path: str, question: str, session_id: str = None) -> Dict[str, Any]:
        """Process a single image-question pair using your VisionLanguageAgent"""
        print(f"\nProcessing query...")
        print(f"Image: {image_path}")
        print(f"Question: {question}")
        
        start_time = time.time()
        
        try:
            # Call the async process_query method properly
            result = await self.agent.process_query(image_path, question, session_id)
            
            processing_time = time.time() - start_time
            
            # Display results using your actual response format
            self.display_results(result, processing_time)
            
            return result
            
        except Exception as e:
            print(f"Error processing query: {e}")
            logger.error(f"Query processing error: {e}")
            return {
                'success': False, 
                'error': str(e),
                'session_id': session_id,
                'processing_time': time.time() - start_time
            }
    
    def display_results(self, result: Dict[str, Any], processing_time: float):
        """Display formatted results matching your VisionLanguageAgent response format"""
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        if result.get('success', False):
            # Final Answer
            print(f"\nFINAL ANSWER:")
            print(f"{result.get('response', 'No response provided')}")
            
            # Vision Analysis Results
            visual_features = result.get('visual_features', {})
            if visual_features:
                print(f"\nVISION ANALYSIS:")
                
                # Objects detected
                if 'detect_objects' in visual_features:
                    obj_data = visual_features['detect_objects']
                    if isinstance(obj_data, dict) and 'result' in obj_data:
                        objects = obj_data['result'].get('objects', [])
                        print(f"Objects Detected ({len(objects)}):")
                        for obj in objects[:5]:  # Show top 5
                            class_name = obj.get('class', 'unknown')
                            confidence = obj.get('confidence', 0)
                            print(f"  • {class_name} (confidence: {confidence:.2f})")
                
                # Text found
                if 'read_text' in visual_features:
                    text_data = visual_features['read_text']
                    if isinstance(text_data, dict) and 'result' in text_data:
                        full_text = text_data['result'].get('full_text', '').strip()
                        if full_text:
                            print(f"Text Found: '{full_text}'")
                
                # Scene description
                if 'describe_scene' in visual_features:
                    scene_data = visual_features['describe_scene']
                    if isinstance(scene_data, dict) and 'result' in scene_data:
                        description = scene_data['result'].get('description', '')
                        if description:
                            print(f"Scene Description: {description}")
            
            # Reasoning Chain
            reasoning_chain = result.get('reasoning_chain', [])
            if reasoning_chain:
                print(f"\nREASONING CHAIN:")
                for i, step in enumerate(reasoning_chain, 1):
                    if isinstance(step, dict):
                        reasoning_text = step.get('reasoning', 'No reasoning provided')
                        # Truncate long reasoning for display
                        if len(reasoning_text) > 100:
                            reasoning_text = reasoning_text[:100] + "..."
                        print(f"Step {i}: {reasoning_text}")
                        
                        # Show evidence if available
                        evidence = step.get('evidence', [])
                        if evidence:
                            print(f"         Evidence: {', '.join(evidence[:2])}")
                            if len(evidence) > 2:
                                print(f"         ... and {len(evidence) - 2} more")
            
            # Performance Stats
            print(f"\nPERFORMANCE:")
            print(f"Processing Time: {result.get('processing_time', processing_time):.2f}s")
            print(f"Session ID: {result.get('session_id', 'N/A')}")
            
        else:
            # Error case
            print(f"Query failed: {result.get('error', 'Unknown error')}")
            if 'session_id' in result:
                print(f"Session ID: {result['session_id']}")
        
        print("="*70)
    
    async def interactive_mode(self):
        """Interactive conversation mode"""
        print("\nStarting Interactive Mode")
        print("Commands: 'quit' to exit, 'new' for new session, 'stats' for system stats")
        
        self.current_session_id = None
        
        while True:
            try:
                # Get image path
                print(f"\nEnter image path (or command): ", end="")
                image_input = input().strip()
                
                if image_input.lower() == 'quit':
                    break
                elif image_input.lower() == 'new':
                    self.current_session_id = None
                    print("Started new session")
                    continue
                elif image_input.lower() == 'stats':
                    self.display_system_statistics()
                    continue
                
                # Validate image path
                if not os.path.exists(image_input):
                    print(f"Image not found: {image_input}")
                    continue
                
                # Get question
                print("Enter your question: ", end="")
                question = input().strip()
                
                if not question:
                    print("Question cannot be empty")
                    continue
                
                # Process query (now properly awaiting the async method)
                result = await self.process_single_query(
                    image_input, question, self.current_session_id
                )
                
                # Update session ID for conversation continuity
                if result.get('success'):
                    self.current_session_id = result.get('session_id')
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                logger.error(f"Interactive mode error: {e}")
    
    def display_system_statistics(self):
        """Display comprehensive system statistics"""
        try:
            stats = self.agent.get_system_stats()
            
            print("\nSYSTEM STATISTICS")
            print("="*50)
            
            # System stats
            system_stats = stats.get('system', {})
            print(f"Total Queries: {system_stats.get('total_queries', 0)}")
            print(f"Successful: {system_stats.get('successful_responses', 0)}")
            print(f"Errors: {system_stats.get('error_count', 0)}")
            print(f"Avg Response Time: {system_stats.get('average_response_time', 0):.2f}s")
            
            # Cache stats
            cache_stats = stats.get('cache', {})
            print(f"\nCache Performance:")
            print(f"Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
            print(f"Total Requests: {cache_stats.get('total_requests', 0)}")
            print(f"Memory Cache Size: {cache_stats.get('memory_cache_size', 0)}")
            
            # Vision model stats
            vision_stats = stats.get('vision_router', {})
            if vision_stats:
                print(f"\nVision Model Performance:")
                for model, model_stats in vision_stats.items():
                    print(f"{model}:")
                    print(f"  Calls: {model_stats.get('total_calls', 0)}")
                    print(f"  Cache Hit Rate: {model_stats.get('cache_hit_rate', 0):.1%}")
                    print(f"  Avg Time: {model_stats.get('avg_processing_time', 0):.3f}s")
                    print(f"  Status: {model_stats.get('status', 'unknown')}")
            
        except Exception as e:
            print(f"Error getting stats: {e}")
    
    async def batch_process(self, image_folder: str, questions_file: str):
        """Process multiple images with questions from file"""
        print(f"\nBatch Processing Mode")
        print(f"Images folder: {image_folder}")
        print(f"Questions file: {questions_file}")
        
        # Load questions
        try:
            with open(questions_file, 'r') as f:
                questions = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Questions file not found: {questions_file}")
            return
        
        # Get image files
        image_folder = Path(image_folder)
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in image_folder.glob('*') 
                      if f.suffix.lower() in supported_extensions]
        
        if not image_files:
            print(f"No supported images found in {image_folder}")
            print(f"Supported formats: {', '.join(supported_extensions)}")
            return
        
        print(f"Processing {len(image_files)} images with {len(questions)} questions...")
        
        results = []
        total_queries = len(image_files) * len(questions)
        current_query = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n--- Image {i}/{len(image_files)}: {image_path.name} ---")
            
            for j, question in enumerate(questions, 1):
                current_query += 1
                print(f"[{current_query}/{total_queries}] {question}")
                
                result = await self.process_single_query(str(image_path), question)
                
                results.append({
                    'image': str(image_path),
                    'question': question,
                    'result': result,
                    'timestamp': time.time()
                })
                
                # Brief pause between queries to avoid overwhelming the system
                time.sleep(0.5)
        
        # Save results
        output_file = f"batch_results_{int(time.time())}.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nBatch processing complete! Results saved to: {output_file}")
            
            # Summary statistics
            successful = sum(1 for r in results if r['result'].get('success'))
            print(f"Summary: {successful}/{len(results)} queries successful")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    async def demo_mode(self):
        """Run demonstration with complex reasoning examples"""
        print("\nDemo Mode - Complex Reasoning Examples")
        print("This mode showcases your system's advanced reasoning capabilities")
        
        demo_scenarios = [
            {
                'name': 'Object Relationship Analysis',
                'question': 'What is the relationship between the objects in this image?',
                'description': 'Tests spatial and contextual understanding'
            },
            {
                'name': 'Safety Assessment', 
                'question': 'What safety concerns do you see in this scene?',
                'description': 'Tests risk analysis and contextual reasoning'
            },
            {
                'name': 'Predictive Reasoning',
                'question': 'What might happen next in this scene?',
                'description': 'Tests temporal reasoning and prediction'
            },
            {
                'name': 'Contextual Understanding',
                'question': 'Why might the person be performing this action?',
                'description': 'Tests motivation and intent inference'
            },
            {
                'name': 'Multi-step Analysis',
                'question': 'Describe the scene and explain what led to this situation',
                'description': 'Tests comprehensive scene understanding'
            }
        ]
        
        print("Enter image path for demo: ", end="")
        image_path = input().strip()
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        
        session_id = None
        
        for i, scenario in enumerate(demo_scenarios, 1):
            print(f"\n{'='*60}")
            print(f"DEMO {i}: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            print(f"Question: {scenario['question']}")
            print('='*60)
            
            result = await self.process_single_query(
                image_path, scenario['question'], session_id
            )
            
            if result.get('success'):
                session_id = result['session_id']  # Continue conversation
            
            if i < len(demo_scenarios):
                input("\nPress Enter to continue to next demo...")
    
    def cleanup(self):
        """Cleanup resources using your agent's cleanup method"""
        try:
            if self.agent:
                self.agent.cleanup()
            print("Cleanup completed")
        except Exception as e:
            print(f"Cleanup error: {e}")


async def main_async():
    """Async main entry point"""
    parser = argparse.ArgumentParser(
        description='Vision-Language Agent Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py --mode single --image img.jpg --question "What do you see?"
  python main.py --mode batch --image-folder ./images --questions-file questions.txt
  python main.py --mode demo                        # Complex reasoning demo
        """
    )
    
    parser.add_argument('--mode', choices=['interactive', 'single', 'batch', 'demo'], 
                       default='interactive', help='Operation mode (default: interactive)')
    parser.add_argument('--image', help='Image path for single mode')
    parser.add_argument('--question', help='Question for single mode')
    parser.add_argument('--image-folder', help='Image folder for batch mode')
    parser.add_argument('--questions-file', help='Questions file for batch mode')
    parser.add_argument('--session-id', help='Session ID to continue conversation')
    
    args = parser.parse_args()
    
    # Initialize interface
    interface = VisionAgentInterface()
    interface.print_banner()
    
    try:
        if args.mode == 'interactive':
            await interface.interactive_mode()
            
        elif args.mode == 'single':
            if not args.image or not args.question:
                print("Single mode requires --image and --question arguments")
                parser.print_help()
                return
            await interface.process_single_query(args.image, args.question, args.session_id)
            
        elif args.mode == 'batch':
            if not args.image_folder or not args.questions_file:
                print("Batch mode requires --image-folder and --questions-file arguments")
                parser.print_help()
                return
            await interface.batch_process(args.image_folder, args.questions_file)
            
        elif args.mode == 'demo':
            await interface.demo_mode()
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        interface.cleanup()


def main():
    """Synchronous wrapper for async main"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()