"""
/agent/reasoning.py
Reasoning engine for the financial agent system.

This module implements the decision-making capabilities of the agent,
including action selection, tool parameter determination, and error handling.
It uses a ReAct-style reasoning approach to determine the next best action.
"""

import logging
from typing import Dict, List, Any, Optional
from utils.llm_utils import LLMManager 
import json
import time


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Engine responsible for agent reasoning and decision-making.
    
    This class implements ReAct-style reasoning to determine the agent's next action
    based on the current context, available tools, and memory.
    """
    
    def __init__(self, llm_manager: LLMManager, config: Dict[str, Any] = None):
        """
        Initialize the reasoning engine with optional configurations.
        
        Args:
            config: Optional configuration parameters for the reasoning engine
        """

        self.config = config or {}
        self.llm_manager = llm_manager
        self.reasoning_template = self.config.get("reasoning_template", self._default_reasoning_template)
        self.failure_handling_template = self.config.get("failure_template", self._default_failure_template)
        self.summary_template = self.config.get("summary_template", self._default_summary_template)
        logger.info("Reasoning engine initialized")
    
          
    def _default_reasoning_template(self) -> str:
            """Default template for reasoning about the next action."""
            return """
            # Financial Agent Task Reasoning

            ## Current Task
            {task}

            ## Current Step
            {step} of {max_steps}

            ## Available Tools
            {tools_description}

            ## Recent Context
            {context_summary}

            ## Current State
            {state_summary}

            ## Step-by-Step Reasoning
            Think carefully about what you need to do next to progress toward completing the task.
            Consider what information you have, what information you need, and which tool would be most appropriate.

            1.
            2.
            3.

            ## Decision
            Based on the above reasoning, I should:

            ## Action JSON
            Provide your selected action in the following JSON format:
            # CORRECTED: Double braces for literal examples
            - For tool use: {{"type": "tool", "tool_name": "name_of_tool", "parameters": {{"param1": "value1", "param2": "value2"}}}}
            - To finish the task: {{"type": "finish", "reason": "explanation"}}

            ```json
            {{
            }}
            ```
            """

    
    
    def _default_failure_template(self) -> str:
        """Default template for handling tool execution failures."""
        return """
        # Error Recovery Reasoning

        ## Failed Action
        Tool: {tool_name}
        Parameters: {parameters}

        ## Error Details
        {error_message}

        ## Available Tools
        {tools_description}

        ## Recent Context
        {context_summary}

        ## Step-by-Step Recovery Planning
        Think carefully about why the tool might have failed and what to do next.
        Consider whether to retry with different parameters, use a different tool, or abort the task.

        1.
        2.
        3.

        ## Decision
        Based on the above reasoning, I should:

        ## Recovery Action JSON
        Provide your recovery action in the following JSON format:
        # CORRECTED: Double braces for literal examples
        - To retry with the same tool: {{"type": "retry", "tool_name": "name_of_tool", "parameters": {{"param1": "value1"}}}}
        - To try a different tool: {{"type": "tool", "tool_name": "different_tool", "parameters": {{"param1": "value1"}}}}
        - To abort the task: {{"type": "abort", "reason": "explanation"}}

        ```json
        {{
        }}
        ```
        """
    
    def _default_summary_template(self) -> str:
        """Default template for generating task summaries."""
        return """
        # Task Summary
        
        ## Original Task
        {task}
        
        ## Actions Taken
        {actions_summary}
        
        ## Results
        {results_summary}
        
        ## Provide a concise summary of the task execution
        - Key findings or insights
        - Successful actions
        - Any issues encountered
        - Final outcome
        
        Summary:
        """
    
    def _format_tools_description(self, tools: Dict[str, Dict[str, Any]]) -> str:
        """
        Format the available tools into a readable description.
        
        Args:
            tools: Dictionary of tool metadata
            
        Returns:
            Formatted string describing available tools
        """
        description = []
        for name, info in tools.items():
            params = info.get("parameters", {})
            params_desc = ", ".join([f"{p}" for p in params]) if params else "None"
            description.append(f"- {name}: {info['description']} (Parameters: {params_desc})")
        
        return "\n".join(description)
    
    def _format_context_summary(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format recent memories into a readable context summary.
        
        Args:
            memories: List of memory items
            
        Returns:
            Formatted string summarizing recent context
        """
        if not memories:
            return "No recent context available."
        
        summary = []
        for i, memory in enumerate(memories[-5:]):  # Only use the 5 most recent memories
            if memory["type"] == "tool_execution":
                summary.append(f"{i+1}. Used tool '{memory['tool']}' with parameters {memory['parameters']}")
                summary.append(f"   Result: {str(memory['result'])[:200]}{'...' if len(str(memory['result'])) > 200 else ''}")
            elif memory["type"] == "tool_execution_error":
                summary.append(f"{i+1}. ERROR using tool '{memory['tool']}': {memory['error']}")
            elif memory["type"] == "task_start":
                summary.append(f"{i+1}. Started task: {memory['description']}")
        
        return "\n".join(summary)
    
    def _format_state_summary(self, state: Dict[str, Any]) -> str:
        """
        Format the current state into a readable summary.
        
        Args:
            state: Current agent state
            
        Returns:
            Formatted string summarizing current state
        """
        summary = [
            f"Task status: {state['task_status']}",
            f"Tools used: {dict(sorted(state['session_stats']['tools_used'].items()))}"
        ]
        
        if state.get("goal_progress"):
            summary.append("Goal progress:")
            for goal, progress in state["goal_progress"].items():
                summary.append(f"- {goal}: {progress}")
        
        return "\n".join(summary)
    
    def decide_next_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide on the next action based on the current context.
        
        Args:
            context: Current context including task, memory, tools, etc.
            
        Returns:
            Dictionary representing the next action to take
        """ 
        
        # Prepare the context for the reasoning prompt
        tools_description = self._format_tools_description(context["available_tools"])
        context_summary = self._format_context_summary(context["short_term_memory"])
        state_summary = self._format_state_summary(context["current_state"])
        
        # Create the reasoning prompt
        prompt = self.reasoning_template().format(
            task=context["task"],
            step=context["step"],
            max_steps=context["max_steps"],
            tools_description=tools_description,
            context_summary=context_summary,
            state_summary=state_summary
        )
        
        # Get the reasoning response from the LLM
        logger.debug("Sending reasoning prompt to LLM")
        llm_response = self.llm_manager.query(prompt)

        time.sleep(40)
        
        # Extract the JSON action from the response
        try:
            # Look for JSON between triple backticks
            json_str = self._extract_json(llm_response)
            action = json.loads(json_str)
            
            # Validate the action format
            if action["type"] not in ["tool", "finish"]:
                logger.warning(f"Invalid action type: {action['type']}")
                # Default to a safe action
                return {
                    "type": "finish", 
                    "reason": "Invalid action type returned by reasoning"
                }
                
            logger.info(f"Decided on action: {action['type']}")
            return action
            
        except Exception as e:
            logger.error(f"Error parsing action JSON: {str(e)}")
            # Return a default safe action
            return {
                "type": "finish",
                "reason": f"Failed to determine next action: {str(e)}"
            }
    
    def handle_tool_failure(self, context: Dict[str, Any], failed_action: Dict[str, Any], 
                          failure_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a tool execution failure by deciding on a recovery action.
        
        Args:
            context: Current context dictionary
            failed_action: The action that failed
            failure_result: Details about the failure
            
        Returns:
            Dictionary representing the recovery action to take
        """
        # Prepare the context for the failure handling prompt
        tools_description = self._format_tools_description(context["available_tools"])
        context_summary = self._format_context_summary(context["short_term_memory"])
        
        # Create the failure handling prompt
        prompt = self.failure_handling_template().format(
            tool_name=failed_action["tool_name"],
            parameters=json.dumps(failed_action.get("parameters", {}), indent=2),
            error_message=failure_result["error"],
            tools_description=tools_description,
            context_summary=context_summary
        )
        
        # Get the failure handling response from the LLM
        logger.debug("Sending failure handling prompt to LLM")
        llm_response = self.llm_manager.query(prompt)
        time.sleep(40)
        
        # Extract the JSON recovery action from the response
        try:
            # Look for JSON between triple backticks
            json_str = self._extract_json(llm_response)
            recovery_action = json.loads(json_str)
            
            # Validate the recovery action format
            if recovery_action["type"] not in ["retry", "tool", "abort"]:
                logger.warning(f"Invalid recovery action type: {recovery_action['type']}")
                # Default to abort if the recovery action is invalid
                return {"type": "abort", "reason": "Invalid recovery action type"}
                
            logger.info(f"Decided on recovery action: {recovery_action['type']}")
            return recovery_action
            
        except Exception as e:
            logger.error(f"Error parsing recovery action JSON: {str(e)}")
            # Default to abort if we can't parse the recovery action
            return {
                "type": "abort",
                "reason": f"Failed to determine recovery action: {str(e)}"
            }
    
    def generate_task_summary(self, task: str, memories: List[Dict[str, Any]], 
                            results: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of the task execution.
        
        Args:
            task: Original task description
            memories: List of memory items from the task execution
            results: List of tool execution results
            
        Returns:
            String summarizing the task execution
        """
       
        actions = []
        for memory in memories:
            if memory["type"] == "tool_execution":
                actions.append(f"- Used {memory['tool']} with parameters {memory['parameters']}")
        actions_summary = "\n".join(actions) if actions else "No actions were taken."
        
        
        result_items = []
        for result in results:
            if result["status"] == "success":
                summary = str(result["result"])
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                result_items.append(f"- {result['tool']} completed successfully: {summary}")
            else:
                result_items.append(f"- {result['tool']} failed: {result['error']}")
        results_summary = "\n".join(result_items) if result_items else "No results to report."
        
        
        prompt = self.summary_template().format(
            task=task,
            actions_summary=actions_summary,
            results_summary=results_summary
        )
        
        
        logger.debug("Sending summary generation prompt to LLM")
        summary = self.llm_manager.query(prompt)
        time.sleep(40)
        
        
        try:
            summary_text = summary.split("Summary:")[-1].strip()
            return summary_text
        except:
            return summary  # Return the whole thing if we can't parse it
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON string from text, looking for content between ```json and ```.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted JSON string
            
        Raises:
            ValueError: If no JSON is found in the text
        """
        # Try to find JSON content between ```json and ```
        if "```json" in text:
            parts = text.split("```json")
            if len(parts) > 1:
                json_part = parts[1].split("```")[0].strip()
                return json_part
        
        # Fallback: Try to find content between any ``` and ```
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 2:
                # Use the content of the first code block
                return parts[1].strip()
        
        # Second fallback: Look for content that looks like JSON (between { and })
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        raise ValueError("No JSON content found in the response")
