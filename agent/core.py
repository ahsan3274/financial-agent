"""
/agent/.core.py
Core agent functionality for the financial agent system.

This module implements the core agent capabilities, including the main agent loop,
tool management, state tracking, and coordination between different components.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
import time

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from agent.reasoning import ReasoningEngine
from collections import defaultdict  
from utils.llm_utils import LLMManager
#from typing import DefaultDict  


#Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class FinancialAgent:
    """Main agent class that orchestrates the financial management system."""
    
    def __init__(self,
                 tools: Dict[str, Dict],
                 llm_manager: LLMManager, # Pass LLMManager instance
                 long_term_memory: LongTermMemory, # Pass LongTermMemory instance
                 config: Dict[str, Any] = None):
        """
        Initialize the financial agent with tools, dependencies, and optional configuration.

        Args:
            tools: Dictionary of tool names mapped to their implementation functions.
            llm_manager: An initialized instance of LLMManager.
            long_term_memory: An initialized instance of LongTermMemory.
            config: Optional configuration parameters for the agent.
        """
        self.config = config or {
        "max_retries": 3,
        "memory_consolidation_interval": 5,
        "llm_timeout": 30,
        "safety_guardrails": {
            "max_budget_override": 1000,
            "allowed_tool_categories": ["financial"]
        }
    }
        self.tools = tools
        self.llm_manager = llm_manager           
        self.long_term_memory = long_term_memory 
        # Initialize components that depend on others *after* dependencies are stored
        self.short_term_memory = ShortTermMemory(max_items=self.config.get("short_term_memory_size", 10))
        self.reasoning_engine = ReasoningEngine(llm_manager=self.llm_manager, config=self.config.get("reasoning_config")) # PASS llm_manager
        self.state = {
            "current_task": None,
            "task_status": "idle",
            "active_goals": [],
            "completed_actions": [],
            "tool_usage_history": {}, 
            "error_context": None,
            "session_stats": {
                "start_time": time.time(),
                "tools_used": defaultdict(int),
                "llm_calls": 0, 
                "success_rate": 1.0, 
                "successful_actions": 0,
                "failed_actions": 0
            }
        }
        logger.info(f"Financial agent initialized with {len(tools)} tools.")
        logger.info("Financial agent initialized with %d tools", len(tools))
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """
        Update the agent's current state with new information.
        
        Args:
            updates: Dictionary containing state updates
        """
        for key, value in updates.items():
            if key in self.state:
                if isinstance(self.state[key], dict) and isinstance(value, dict):
                    self.state[key].update(value)
                else:
                    self.state[key] = value
            else:
                self.state[key] = value

    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of available tools with their descriptions.
        
        Returns:
            Dictionary mapping tool names to their metadata
        """
        tool_info = {}
        for tool_name, tool_meta in self.tools.items():
            tool_info[tool_name] = {
                "description": tool_meta["description"],
                "parameters": tool_meta["parameters"]
            }
        return tool_info
       
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a specific tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool
            
        Returns:
            Tool execution results
            
        Raises:
            ValueError: If the tool doesn't exist

        """
        
        tool_meta = self.tools.get(tool_name)

        if not tool_meta:
            raise ValueError(f"Tool {tool_name} not registered")
    
        
        missing_params = []
        if missing_params:
            logger.error(f"Tool '{tool_name}' called with missing parameters: {missing_params}")
            return {
                "status": "error", # <<< ADD THIS KEY
                "tool": tool_name, # <<< ADD Tool name for context
                "error": f"Missing required parameters: {missing_params}",
                "parameters": kwargs # <<< Include provided (incomplete) params
            }
            
        
        self.state["last_action"] = {
            "tool": tool_name,
            "params": kwargs,
            "timestamp": time.time()
        }
        
        try:
            logger.info(f"Executing tool: {tool_name}")
            # Track tool usage in session stats
            self.state["session_stats"]["tools_used"][tool_name] = self.state["session_stats"]["tools_used"].get(tool_name, 0) + 1
            
            # Execute the tool
            start_time = time.time()
            result = tool_meta["function"](**kwargs)
            execution_time = time.time() - start_time
            
            
            execution_result = {
                "status": "success",
                "tool": tool_name,
                "result": result,
                "execution_time": execution_time
            }
            
            
            self.state["session_stats"]["successful_actions"] += 1
            self.state["last_tool_result"] = execution_result
            
            
            self.short_term_memory.add_item({
                "type": "tool_execution",
                "tool": tool_name,
                "parameters": kwargs,
                "result": result,
                "timestamp": time.time()
            })
            
            logger.info(f"Tool {tool_name} executed successfully in {execution_time:.2f}s")
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            error_result = {
                "status": "error",
                "tool": tool_name,
                "error": str(e),
                "parameters": kwargs
            }
            self.state["session_stats"]["failed_actions"] += 1
            self.state["last_tool_result"] = error_result
            
            # Record the failed action in short-term memory
            self.short_term_memory.add_item({
                "type": "tool_execution_error",
                "tool": tool_name,
                "parameters": kwargs,
                "error": str(e),
                "timestamp": time.time()
            })
            
            return error_result
    
    def run(self, task_description: str, max_steps: int = 10) -> Dict[str, Any]:
        """
        Run the agent to complete a financial task.
        
        Args:
            task_description: Description of the task to complete
            max_steps: Maximum number of steps to take before stopping
            
        Returns:
            Dictionary containing the task results and execution summary
        """
        #Add goal tracking
        self.state["active_goals"].append({
            "description": task_description,
            "created_at": time.time(),
            "progress": []
        })

        
        retry_attempts = 0
        max_retries = self.config.get("max_retries", 3)

        logger.info(f"Starting task: {task_description}")
        self.update_state({
            "current_task": task_description,
            "task_status": "in_progress",
            "last_tool_result": None
        })
        
        # Add the task to short-term memory
        self.short_term_memory.add_item({
            "type": "task_start",
            "description": task_description,
            "timestamp": time.time()
        })
        
        
        relevant_memories = self.long_term_memory.search(task_description)
        if relevant_memories:
            logger.info(f"Found {len(relevant_memories)} relevant memories for this task")
            
        steps_taken = 0
        results = []
        
        while steps_taken < max_steps and self.state["task_status"] == "in_progress":
            # Add progress tracking
            self.state["current_progress"] = {
                "steps": steps_taken,
                "latest_result": results[-1] if results else None
            }
        
           
            if self.state.get("error_context"):
                context["error_context"] = self.state["error_context"]
            # Get the next action from the reasoning engine
            context = {
                "task": task_description,
                "short_term_memory": self.short_term_memory.get_items(),
                "relevant_long_term_memory": relevant_memories,
                "available_tools": self.get_available_tools(),
                "current_state": self.state,
                "step": steps_taken + 1,
                "max_steps": max_steps
            }
            
            action = self.reasoning_engine.decide_next_action(context)
            
            if action["type"] == "finish":
                logger.info("Agent decided to finish the task")
                self.update_state({"task_status": "completed"})
                break
                
            elif action["type"] == "tool":
                # Execute the chosen tool
                tool_result = self.execute_tool(
                    action["tool_name"], 
                    **action.get("parameters", {})
                )
                results.append(tool_result)
                #  # -------------- ---
                # print("-" * 40) # Separator for clarity
                # print(f"DEBUG: Tool Executed: {action.get('tool_name')}")
                # print(f"DEBUG: Tool Parameters: {action.get('parameters', {})}")
                # print(f"DEBUG: Raw Tool Result Received by run(): {tool_result}")
                # print(f"DEBUG: Type of Tool Result: {type(tool_result)}")
                # if isinstance(tool_result, dict):
                #      print(f"DEBUG: Keys in Tool Result: {tool_result.keys()}")
                # print("-" * 40) # 
                # # ---------------------------------
                # If the tool failed, decide whether to retry, use an alternative, or abort
                if tool_result["status"] == "error" and steps_taken < max_steps - 1:
                    recovery_action = self.reasoning_engine.handle_tool_failure(
                        context, 
                        action, 
                        tool_result
                    )
                    
                    if recovery_action["type"] == "retry":
                        logger.info(f"Retrying failed tool: {action['tool_name']}")
                        # Note: The retry will happen in the next loop iteration
                        
                    elif recovery_action["type"] == "abort":
                        logger.warning("Aborting task due to unrecoverable error")
                        self.update_state({"task_status": "failed"})
                        break
                        
            steps_taken += 1
            
        
        if steps_taken >= max_steps and self.state["task_status"] == "in_progress":
            logger.warning(f"Reached maximum steps ({max_steps}) without completing the task")
            self.update_state({"task_status": "incomplete"})
            
        
        task_summary = self.reasoning_engine.generate_task_summary(
            task_description, 
            self.short_term_memory.get_items(), 
            results
        )
        
        
        self.long_term_memory.store({
            "type": "task_execution",
            "task": task_description,
            "steps": steps_taken,
            "status": self.state["task_status"],
            "summary": task_summary,
            "timestamp": time.time()
        })
        
        execution_summary = {
            "task": task_description,
            "status": self.state["task_status"],
            "steps_taken": steps_taken,
            "results": results,
            "summary": task_summary,
            "execution_time": time.time() - self.state["session_stats"]["start_time"]
        }
        
        logger.info(f"Task completed with status: {self.state['task_status']}")

        return execution_summary

        
        
    def reset(self) -> None:
        """Reset the agent's short-term memory and state."""
        self.short_term_memory.clear()
        self.state = {
            "current_task": None,
            "task_status": "idle",
            "active_goals": [],
            "last_tool_result": None,
            "goal_progress": {},
            "session_stats": {
                "tools_used": {},
                "start_time": time.time(),
                "successful_actions": 0,
                "failed_actions": 0
            }
        }
        logger.info("Agent state and short-term memory reset")

    def _consolidate_memory(self):
        """Transfer important short-term memories to long-term storage"""
        important_events = [
            item for item in self.short_term_memory.get_items()
            if item.get("type") in ["tool_execution", "task_complete"]
        ]
        
        for event in important_events:
            self.long_term_memory.store(event)
        
        self.short_term_memory.clear()

    def add_tool(self, tool_name: str, tool_func: Callable, params: dict):
        """Dynamically register new tools"""
        self.tools[tool_name] = {
            "function": tool_func,
            "parameters": params,
            "description": tool_func.__doc__
        }

    def get_agent_status(self) -> dict:
        """Return current agent state for monitoring"""
        return {
            "uptime": time.time() - self.state["session_stats"]["start_time"],
            "memory_usage": {
                "short_term": len(self.short_term_memory.get_items()),
                "long_term": self.long_term_memory.size()
            },
            "active_goals": len(self.state["active_goals"]),
            "tool_metrics": self.state["session_stats"]["tools_used"]
        }

    def graceful_shutdown(self):
        """Handle cleanup operations"""
        self._consolidate_memory()
        self.long_term_memory.save_all()
        logger.info("Agent shutdown completed")

    def register_tool_plugin(self, plugin_module):
        """Dynamically load tools from a plugin module"""
        for tool_name in getattr(plugin_module, "EXPORTED_TOOLS", []):
            self.tools[tool_name] = {
                "function": getattr(plugin_module, tool_name),
                "description": getattr(plugin_module, f"{tool_name}_DESC", ""),
                "parameters": getattr(plugin_module, f"{tool_name}_PARAMS", {})
            }   
