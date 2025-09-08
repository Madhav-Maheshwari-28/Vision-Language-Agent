"""
Main interface for the Vision-Language Agent system.
Run this file to test and interact with your complete Vision-Language Agent.
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv(override=True)
# Import your modules
try:
    from main_agent import VisionLanguageAgent
    from data_structures import VisionQuery, VisionResult, ReasoningStep
except ImportError as e:
    print(f"Error importing modules: {e}")
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
    """Main interface for interacting with the Vision-Language Agent"""
    
    def __init__(self):
        self.agent = None
        self.current_session_id = None
        self.setup_environment()
    
    def setup_environment(self):
        """Setup environment and validate requirements"""
        print("🚀 Initializing Vision-Language Agent...")
        
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
            print("   The reasoning engine may not work properly")
            
            # Option to input API key
            api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
        
        # Create necessary directories
        for directory in ['temp_processing', 'cache']:
            Path(directory).mkdir(exist_ok=True)
        
        # Initialize the agent
        try:
            self.agent = VisionLanguageAgent()
            print("✅ Vision-Language Agent initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing agent: {e}")
            sys.exit(1)
    
    def print_banner(self):
        """Print welcome banner"""
        banner = """
╔════════════════════════════════════════════════════════════════╗
║                 Vision-Language Agent v1.0                     ║
║                                                                ║
║  Advanced AI system for complex visual scene understanding     ║
║  • Multi-step reasoning with LLM integration                  ║
║  • Real vision models (YOLO, OCR, BLIP, VideoMAE)            ║
║  • Multi-turn conversational memory                           ║
║  • Intelligent caching system                                 ║
╚════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def print_system_info(self):
        """Display system information and capabilities"""
        print("\n📋 System Capabilities:")
        print("   🔍 Object Detection (YOLO) - Identify objects and their locations")
        print("   📝 Text Recognition (OCR) - Extract text from images")  
        print("   🎨 Scene Description (BLIP) - Generate natural language descriptions")
        print("   🎬 Action Recognition (VideoMAE) - Understand actions in videos")
        print("   🧠 Multi-step Reasoning - Complex question answering with LLM")
        print("   💾 Conversation Memory - Maintain context across turns")
        print("   ⚡ Multi-level Caching - Fast response times")
    
    async def process_single_query(self, image_path: str, question: str, session_id: str = None) -> Dict[str, Any]:
        """Process a single image-question pair"""
        print(f"\n🔄 Processing query...")
        print(f"   📷 Image: {image_path}")
        print(f"   ❓ Question: {question}")
        
        start_time = time.time()
        
        try:
            # Process the query
            result = await self.agent.process_query(image_path, question, session_id)
            
            processing_time = time.time() - start_time
            
            # Display results
            self.display_results(result, processing_time)
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            logger.error(f"Query processing error: {e}")
            return {'success': False, 'error': str(e)}
    
    def display_results(self, result: Dict[str, Any], processing_time: float):
        """Display formatted results"""
        print("\n" + "="*70)
        print("📊 RESULTS")
        print("="*70)
        
        if result.get('success'):
            # Final Answer
            print(f"\n💡 FINAL ANSWER:")
            print(f"   {result['response']}")
            
            # Vision Analysis Results
            if 'visual_features' in result:
                print(f"\n👁️ VISION ANALYSIS:")
                features = result['visual_features']
                
                # Objects detected
                if 'detect_objects' in features:
                    objects = features['detect_objects'].get('result', {}).get('objects', [])
                    print(f"   🎯 Objects Detected ({len(objects)}):")
                    for obj in objects[:5]:  # Show top 5
                        print(f"      • {obj.get('class', 'unknown')} (confidence: {obj.get('confidence', 0):.2f})")
                
                # Text found
                if 'read_text' in features:
                    text = features['read_text'].get('result', {}).get('full_text', '')
                    if text.strip():
                        print(f"   📝 Text Found: '{text.strip()}'")
                
                # Scene description
                if 'describe_scene' in features:
                    description = features['describe_scene'].get('result', {}).get('description', '')
                    if description:
                        print(f"   🎨 Scene: {description}")
            
            # Reasoning Chain
            if 'reasoning_chain' in result and result['reasoning_chain']:
                print(f"\n🧠 REASONING CHAIN:")
                for i, step in enumerate(result['reasoning_chain'], 1):
                    print(f"   Step {i}: {step.get('reasoning', 'No reasoning provided')[:100]}...")
                    if step.get('evidence'):
                        print(f"           Evidence: {', '.join(step['evidence'][:2])}...")
            
            # Performance Stats
            print(f"\n⚡ PERFORMANCE:")
            print(f"   Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"   Session ID: {result.get('session_id', 'N/A')}")
            
        else:
            print(f"❌ Query failed: {result.get('error', 'Unknown error')}")
        
        print("="*70)
    
    async def interactive_mode(self):
        """Interactive conversation mode"""
        print("\n🗣️  Starting Interactive Mode")
        print("   Type 'quit' to exit, 'new' for new session, 'stats' for system stats")
        
        self.current_session_id = None
        
        while True:
            try:
                # Get image path
                print(f"\n📷 Enter image path (or 'quit'): ", end="")
                image_input = input().strip()
                
                if image_input.lower() == 'quit':
                    break
                elif image_input.lower() == 'new':
                    self.current_session_id = None
                    print("🔄 Started new session")
                    continue
                elif image_input.lower() == 'stats':
                    self.display_system_stats()
                    continue
                
                # Validate image path
                if not os.path.exists(image_input):
                    print(f"❌ Image not found: {image_input}")
                    continue
                
                # Get question
                print("❓ Enter your question: ", end="")
                question = input().strip()
                
                if not question:
                    print("❌ Question cannot be empty")
                    continue
                
                # Process query
                result = await self.process_single_query(
                    image_input, question, self.current_session_id
                )
                
                # Update session ID for conversation continuity
                if result.get('success'):
                    self.current_session_id = result.get('session_id')
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                logger.error(f"Interactive mode error: {e}")
    
    def display_system_stats(self):
        """Display comprehensive system statistics"""
        try:
            stats = self.agent.get_system_stats()
            
            print("\n📈 SYSTEM STATISTICS")
            print("="*50)
            
            # System stats
            system_stats = stats.get('system', {})
            print(f"Total Queries: {system_stats.get('total_queries', 0)}")
            print(f"Successful: {system_stats.get('successful_responses', 0)}")
            print(f"Errors: {system_stats.get('error_count', 0)}")
            print(f"Avg Response Time: {system_stats.get('average_response_time', 0):.2f}s")
            
            # Cache stats
            cache_stats = stats.get('cache', {})
            print(f"\nCache Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
            print(f"Memory Cache Size: {cache_stats.get('memory_cache_size', 0)}")
            
            # Vision model stats
            vision_stats = stats.get('vision_router', {})
            print(f"\nVision Model Performance:")
            for model, model_stats in vision_stats.items():
                print(f"  {model}:")
                print(f"    Calls: {model_stats.get('total_calls', 0)}")
                print(f"    Cache Hit Rate: {model_stats.get('cache_hit_rate', 0):.1%}")
                print(f"    Avg Time: {model_stats.get('avg_processing_time', 0):.3f}s")
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    async def batch_process(self, image_folder: str, questions_file: str):
        """Process multiple images with questions from file"""
        print(f"\n📁 Batch Processing Mode")
        print(f"   Images: {image_folder}")
        print(f"   Questions: {questions_file}")
        
        # Load questions
        try:
            with open(questions_file, 'r') as f:
                questions = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"❌ Questions file not found: {questions_file}")
            return
        
        # Get image files
        image_folder = Path(image_folder)
        image_files = [f for f in image_folder.glob('*') 
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
        
        if not image_files:
            print(f"❌ No images found in {image_folder}")
            return
        
        print(f"📊 Processing {len(image_files)} images with {len(questions)} questions...")
        
        results = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n--- Processing Image {i}/{len(image_files)}: {image_path.name} ---")
            
            for j, question in enumerate(questions, 1):
                print(f"Question {j}/{len(questions)}: {question}")
                
                result = await self.process_single_query(str(image_path), question)
                
                results.append({
                    'image': str(image_path),
                    'question': question,
                    'result': result
                })
                
                # Brief pause between queries
                await asyncio.sleep(0.5)
        
        # Save results
        output_file = f"batch_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Batch processing complete! Results saved to: {output_file}")
    
    async def demo_mode(self):
        """Run demonstration with example scenarios"""
        print("\n🎬 Demo Mode - Complex Reasoning Examples")
        
        demo_scenarios = [
            {
                'description': 'Object Relationship Analysis',
                'question': 'What is the relationship between the objects in this image?',
            },
            {
                'description': 'Safety Assessment', 
                'question': 'What safety concerns do you see in this scene?',
            },
            {
                'description': 'Predictive Reasoning',
                'question': 'What might happen next in this scene?',
            },
            {
                'description': 'Contextual Understanding',
                'question': 'Why might the person be doing this action?',
            }
        ]
        
        print("📷 Enter image path for demo: ", end="")
        image_path = input().strip()
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        session_id = None
        
        for i, scenario in enumerate(demo_scenarios, 1):
            print(f"\n🎯 Demo {i}: {scenario['description']}")
            print(f"Question: {scenario['question']}")
            
            result = await self.process_single_query(
                image_path, scenario['question'], session_id
            )
            
            if result.get('success'):
                session_id = result['session_id']  # Continue conversation
            
            input("\nPress Enter to continue to next demo...")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.agent:
            self.agent.cleanup()
        print("\n🧹 Cleanup completed")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Vision-Language Agent Interface')
    parser.add_argument('--mode', choices=['interactive', 'single', 'batch', 'demo'], 
                       default='interactive', help='Operation mode')
    parser.add_argument('--image', help='Image path for single mode')
    parser.add_argument('--question', help='Question for single mode')
    parser.add_argument('--image-folder', help='Image folder for batch mode')
    parser.add_argument('--questions-file', help='Questions file for batch mode')
    
    args = parser.parse_args()
    
    # Initialize interface
    interface = VisionAgentInterface()
    interface.print_banner()
    interface.print_system_info()
    
    try:
        if args.mode == 'interactive':
            await interface.interactive_mode()
            
        elif args.mode == 'single':
            if not args.image or not args.question:
                print("❌ Single mode requires --image and --question arguments")
                return
            await interface.process_single_query(args.image, args.question)
            
        elif args.mode == 'batch':
            if not args.image_folder or not args.questions_file:
                print("❌ Batch mode requires --image-folder and --questions-file arguments")
                return
            await interface.batch_process(args.image_folder, args.questions_file)
            
        elif args.mode == 'demo':
            await interface.demo_mode()
            
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    finally:
        interface.cleanup()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())