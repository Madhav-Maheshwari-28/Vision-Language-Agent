"""
Memory management and reasoning engine for the Vision-Language Agent.
Updated to use OpenAI API for LLM reasoning.
"""

import json
import logging
import sqlite3
import uuid
import openai
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Any

from data_structures import ConversationTurn, ReasoningStep, VisionQuery, VisionResult
from vision_models import VisionQueryRouter
# from dotenv import load_dotenv

# load_dotenv(override=True)
logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages conversation memory and context"""
    
    def __init__(self, db_path: str = "conversation_memory.db"):
        self.db_path = db_path
        self.active_memory = {}  # In-memory storage for active sessions
        self.max_active_turns = 10
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for persistent memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                user_input TEXT NOT NULL,
                image_path TEXT,
                response TEXT NOT NULL,
                reasoning_chain TEXT,
                processing_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, turn_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_metadata (
                session_id TEXT PRIMARY KEY,
                user_preferences TEXT,
                session_start DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_activity DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get conversation context for session"""
        context = {
            'session_id': session_id,
            'active_turns': self.active_memory.get(session_id, []),
            'historical_summary': self.get_historical_summary(session_id),
            'user_preferences': self.get_user_preferences(session_id)
        }
        
        return context
    
    def get_historical_summary(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get historical conversation summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_input, response, timestamp
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{'user_input': r[0], 'response': r[1], 'timestamp': r[2]} for r in results]
    
    def get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Get user preferences for session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_preferences
            FROM session_metadata
            WHERE session_id = ?
        ''', (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return json.loads(result[0])
        return {}
    
    def update(self, session_id: str, user_input: str, response: str, 
               image_path: Optional[str] = None, reasoning_chain: List[ReasoningStep] = None,
               processing_time: float = 0.0):
        """Update conversation memory"""
        turn = ConversationTurn(
            session_id=session_id,
            turn_id=str(uuid.uuid4()),
            user_input=user_input,
            image_path=image_path,
            response=response,
            reasoning_chain=reasoning_chain or [],
            processing_time=processing_time
        )
        
        # Update active memory
        if session_id not in self.active_memory:
            self.active_memory[session_id] = []
        
        self.active_memory[session_id].append(turn)
        
        # Keep only recent turns in active memory
        if len(self.active_memory[session_id]) > self.max_active_turns:
            self.active_memory[session_id] = self.active_memory[session_id][-self.max_active_turns:]
        
        # Store in persistent database
        self.store_turn(turn)
        
        # Update session metadata
        self.update_session_metadata(session_id)
    
    def store_turn(self, turn: ConversationTurn):
        """Store conversation turn in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        reasoning_json = json.dumps([asdict(step) for step in turn.reasoning_chain])
        
        cursor.execute('''
            INSERT OR REPLACE INTO conversations
            (session_id, turn_id, user_input, image_path, response, reasoning_chain, processing_time, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            turn.session_id,
            turn.turn_id,
            turn.user_input,
            turn.image_path,
            turn.response,
            reasoning_json,
            turn.processing_time,
            turn.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def update_session_metadata(self, session_id: str):
        """Update session metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO session_metadata (session_id, last_activity)
            VALUES (?, CURRENT_TIMESTAMP)
        ''', (session_id,))
        
        conn.commit()
        conn.close()


class LLMReasoningEngine:
    """Multi-step reasoning engine with OpenAI API integration"""
    
    def __init__(self, vision_router: VisionQueryRouter, max_reasoning_steps: int = 8, 
                 OPENAI_API_KEY: str = None, model: str = "gpt-4"):
        self.vision_router = vision_router
        self.max_reasoning_steps = max_reasoning_steps
        
        # Configure OpenAI
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
        
        self.model = model
        self.temperature = 0.7
        self.max_tokens = 1000
        
        # System prompts for different reasoning tasks
        self.system_prompts = {
            'initial_analysis': """
            You are an expert vision-language AI assistant analyzing images. 
            Your task is to provide initial analysis based on visual evidence.
            Be precise, factual, and identify key elements that help answer user questions.
            Focus on observable facts rather than speculation.
            """,
            
            'reasoning_step': """
            You are continuing a multi-step reasoning process about an image.
            Build logically on previous reasoning steps and visual evidence.
            If you need more visual information, explicitly state what additional analysis would help.
            Be concise but thorough in your reasoning.
            """,
            
            'vision_query_generation': """
            You are deciding what additional visual analysis is needed to answer a question.
            Based on current analysis, determine if more specific visual queries would help.
            Suggest specific types of vision analysis: object detection, text reading, scene description, or action recognition.
            Be specific about parameters that would be most helpful.
            """,
            
            'final_synthesis': """
            You are synthesizing a final answer based on a complete reasoning chain and visual evidence.
            Provide a clear, confident answer that directly addresses the user's question.
            Reference the visual evidence that supports your conclusion.
            If uncertain about any aspect, acknowledge the limitation clearly.
            """
        }
    
    def reason(self, question: str, context: Dict[str, Any], 
               visual_features: Dict[str, VisionResult]) -> Tuple[str, List[ReasoningStep]]:
        """Main reasoning function with multi-step analysis"""
        
        reasoning_chain = []
        current_step = 1
        
        # Step 1: Initial analysis
        initial_step = self.generate_initial_analysis(question, visual_features)
        reasoning_chain.append(initial_step)
        
        # Multi-step reasoning loop
        while current_step < self.max_reasoning_steps:
            # Generate next reasoning step
            next_step = self.generate_reasoning_step(
                question, reasoning_chain, context, current_step + 1
            )
            
            # Check if additional vision queries are needed
            if self.needs_more_vision_info(next_step, question):
                vision_queries = self.generate_vision_queries(next_step, question)
                
                # Execute vision queries
                for query in vision_queries:
                    try:
                        vision_result = self.vision_router.execute_query(query)
                        next_step.evidence.append(f"Vision Query Result: {vision_result.result}")
                    except Exception as e:
                        logger.error(f"Vision query error: {e}")
            
            # Validate reasoning step
            if self.validate_reasoning_step(next_step, reasoning_chain):
                reasoning_chain.append(next_step)
            else:
                # If validation fails, try to recover
                recovery_step = self.generate_recovery_step(next_step, reasoning_chain)
                reasoning_chain.append(recovery_step)
            
            # Check if reasoning is complete
            if self.is_reasoning_complete(reasoning_chain, question):
                break
            
            current_step += 1
        
        # Generate final response
        final_response = self.synthesize_final_response(question, reasoning_chain)
        
        return final_response, reasoning_chain
    
    def call_openai_api(self, system_prompt: str, user_prompt: str, 
                       temperature: float = None) -> str:
        """Make API call to OpenAI"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            # Fallback to basic response
            return f"Analysis step completed. (API Error: {str(e)})"
    
    def generate_initial_analysis(self, question: str, visual_features: Dict[str, VisionResult]) -> ReasoningStep:
        """Generate initial reasoning step from visual features using OpenAI"""
        
        # Extract key information from visual features
        objects = visual_features.get('detect_objects', VisionResult('', {}, 0, 0)).result.get('objects', [])
        text = visual_features.get('read_text', VisionResult('', {}, 0, 0)).result.get('full_text', '')
        scene = visual_features.get('describe_scene', VisionResult('', {}, 0, 0)).result.get('description', '')
        
        # Prepare user prompt
        user_prompt = f"""
        I need to analyze an image to answer this question: "{question}"
        
        Here's what I can see in the image:
        
        Objects detected: {[obj.get('class', 'unknown') for obj in objects]}
        Text found: "{text}"
        Scene description: "{scene}"
        
        Please provide an initial analysis that helps answer the user's question. 
        Focus on the most relevant visual elements and explain how they relate to the question.
        """
        
        # Call OpenAI API
        reasoning = self.call_openai_api(
            self.system_prompts['initial_analysis'], 
            user_prompt
        )
        
        evidence = [
            f"Objects detected: {[obj.get('class', 'unknown') for obj in objects]}",
            f"Text found: {text}",
            f"Scene description: {scene}"
        ]
        
        return ReasoningStep(
            step_number=1,
            reasoning=reasoning,
            evidence=evidence,
            vision_queries=[],
            confidence=0.8
        )
    
    def generate_reasoning_step(self, question: str, reasoning_chain: List[ReasoningStep], 
                              context: Dict[str, Any], step_number: int) -> ReasoningStep:
        """Generate next step in reasoning chain using OpenAI"""
        
        # Prepare context from previous reasoning
        previous_reasoning = "\n".join([
            f"Step {step.step_number}: {step.reasoning}" 
            for step in reasoning_chain
        ])
        
        current_evidence = []
        for step in reasoning_chain:
            current_evidence.extend(step.evidence)
        
        user_prompt = f"""
        I'm working on answering this question: "{question}"
        
        Here's my reasoning so far:
        {previous_reasoning}
        
        Available evidence:
        {chr(10).join(current_evidence)}
        
        What should be my next step in reasoning? Build logically on what I've established so far.
        If I need more visual information, specify what type of analysis would help.
        """
        
        reasoning = self.call_openai_api(
            self.system_prompts['reasoning_step'], 
            user_prompt
        )
        
        return ReasoningStep(
            step_number=step_number,
            reasoning=reasoning,
            evidence=[],
            vision_queries=[],
            confidence=0.75
        )
    
    def needs_more_vision_info(self, step: ReasoningStep, question: str) -> bool:
        """Determine if additional vision queries are needed using OpenAI"""
        user_prompt = f"""
        Question: "{question}"
        Current reasoning step: "{step.reasoning}"
        
        Do I need additional visual analysis to better answer this question? 
        Respond with "YES" if more visual information would help, or "NO" if current analysis is sufficient.
        """
        
        response = self.call_openai_api(
            self.system_prompts['vision_query_generation'], 
            user_prompt,
            temperature=0.3  # Lower temperature for decision making
        )
        
        return response.upper().startswith("YES")
    
    def generate_vision_queries(self, step: ReasoningStep, question: str) -> List[VisionQuery]:
        """Generate additional vision queries based on reasoning needs using OpenAI"""
        
        user_prompt = f"""
        Question: "{question}"
        Current reasoning: "{step.reasoning}"
        
        What specific visual analysis do I need? Choose from:
        - detect_objects (with parameters like confidence_threshold, include_positions)
        - read_text (with parameters like languages)  
        - describe_scene (with parameters like detail_level: basic/medium/detailed, focus area)
        
        Provide your response as a JSON list of queries with this format:
        [{"query_type": "detect_objects", "params": {"confidence_threshold": 0.3}}]
        """
        
        response = self.call_openai_api(
            self.system_prompts['vision_query_generation'], 
            user_prompt,
            temperature=0.3
        )
        
        try:
            # Parse JSON response
            query_specs = json.loads(response)
            queries = []
            
            for spec in query_specs:
                query = VisionQuery(
                    query_type=spec['query_type'],
                    image_path='current_image',  # Will be set by router
                    params=spec.get('params', {})
                )
                queries.append(query)
            
            return queries
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse vision queries from OpenAI response: {e}")
            return []
    
    def validate_reasoning_step(self, step: ReasoningStep, reasoning_chain: List[ReasoningStep]) -> bool:
        """Validate logical consistency of reasoning step using OpenAI"""
        
        if len(reasoning_chain) == 0:
            return True  # First step is always valid
        
        previous_reasoning = "\n".join([
            f"Step {s.step_number}: {s.reasoning}" 
            for s in reasoning_chain
        ])
        
        user_prompt = f"""
        Previous reasoning:
        {previous_reasoning}
        
        New reasoning step: "{step.reasoning}"
        
        Is this new step logically consistent with the previous reasoning? 
        Does it build appropriately on what was established before?
        Respond with "VALID" if consistent, or "INVALID" if there are contradictions or logical issues.
        """
        
        response = self.call_openai_api(
            "You are validating logical consistency in a reasoning chain. Be strict about contradictions.",
            user_prompt,
            temperature=0.2
        )
        
        return response.upper().startswith("VALID")
    
    def generate_recovery_step(self, failed_step: ReasoningStep, reasoning_chain: List[ReasoningStep]) -> ReasoningStep:
        """Generate recovery step when validation fails using OpenAI"""
        
        previous_reasoning = "\n".join([
            f"Step {s.step_number}: {s.reasoning}" 
            for s in reasoning_chain
        ])
        
        user_prompt = f"""
        My previous reasoning:
        {previous_reasoning}
        
        I made an error in my next reasoning step: "{failed_step.reasoning}"
        
        Please provide a corrected reasoning step that builds properly on my previous analysis.
        """
        
        reasoning = self.call_openai_api(
            "You are correcting a reasoning error. Provide a logical next step.",
            user_prompt
        )
        
        return ReasoningStep(
            step_number=failed_step.step_number,
            reasoning=reasoning,
            evidence=["Corrected reasoning step after validation failure"],
            vision_queries=[],
            confidence=0.6
        )
    
    def is_reasoning_complete(self, reasoning_chain: List[ReasoningStep], question: str) -> bool:
        """Check if reasoning chain provides complete answer using OpenAI"""
        
        if len(reasoning_chain) < 2:
            return False
        
        reasoning_summary = "\n".join([
            f"Step {step.step_number}: {step.reasoning}" 
            for step in reasoning_chain
        ])
        
        user_prompt = f"""
        Original question: "{question}"
        
        My reasoning so far:
        {reasoning_summary}
        
        Have I sufficiently answered the original question? 
        Respond with "COMPLETE" if the question is adequately answered, or "CONTINUE" if more reasoning is needed.
        """
        
        response = self.call_openai_api(
            "You are determining if a reasoning chain adequately answers a question.",
            user_prompt,
            temperature=0.2
        )
        
        return response.upper().startswith("COMPLETE")
    
    def synthesize_final_response(self, question: str, reasoning_chain: List[ReasoningStep]) -> str:
        """Generate final response from reasoning chain using OpenAI"""
        
        reasoning_summary = "\n".join([
            f"Step {step.step_number}: {step.reasoning}" 
            for step in reasoning_chain
        ])
        
        all_evidence = []
        for step in reasoning_chain:
            all_evidence.extend(step.evidence)
        
        user_prompt = f"""
        Original question: "{question}"
        
        My complete reasoning process:
        {reasoning_summary}
        
        Supporting evidence:
        {chr(10).join(all_evidence)}
        
        Please provide a clear, direct answer to the original question based on this analysis.
        Reference the key visual evidence that supports your conclusion.
        """
        
        response = self.call_openai_api(
            self.system_prompts['final_synthesis'], 
            user_prompt
        )
        
        return response