from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage
import logging
import os
import json
from typing import Any, Dict, List, Optional, Union
from opencensus.ext.azure.log_exporter import AzureLogHandler
from dotenv import load_dotenv
from datetime import datetime
import traceback

load_dotenv()

# Create a dedicated logger for App Insights
app_insights_logger = logging.getLogger("ARB_Chatbot_AppInsights")
app_insights_logger.setLevel(logging.INFO)

# Prevent duplicate handlers
if not app_insights_logger.handlers:
    try:
        connection_string = os.getenv("App_Insight_Conn_String")
        if connection_string:
            handler = AzureLogHandler(connection_string=connection_string)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            app_insights_logger.addHandler(handler)
            app_insights_logger.propagate = False
            print(f"Azure App Insights handler configured successfully")
        else:
            print("Warning: App_Insight_Conn_String not found in environment variables")
    except Exception as e:
        print(f"Failed to configure Azure App Insights handler: {e}")

class AppInsightsHandler(BaseCallbackHandler):
    """Enhanced AppInsights callback handler for LangChain operations"""
    
    def __init__(self):
        super().__init__()
        self.session_data = {}
        print("AppInsightsHandler initialized")
    
    def _safe_serialize(self, obj: Any) -> str:
        """Safely serialize objects to JSON string"""
        try:
            if isinstance(obj, (str, int, float, bool)):
                return str(obj)
            elif isinstance(obj, dict):
                return json.dumps(obj, default=str, ensure_ascii=False)
            elif isinstance(obj, (list, tuple)):
                return json.dumps(list(obj), default=str, ensure_ascii=False)
            else:
                return str(obj)
        except Exception as e:
            return f"Serialization error: {str(e)}"
    
    def _get_custom_dimensions(self, **kwargs) -> Dict[str, Any]:
        """Create custom dimensions dictionary for App Insights"""
        dimensions = {
            "timestamp": datetime.now().isoformat(),
            "component": "ARB_Chatbot",
            "callback_handler": "AppInsightsHandler"
        }
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ['self', 'kwargs']:
                dimensions[key] = self._safe_serialize(value)
        
        return dimensions

    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Called when LLM starts running"""
        try:
            print("🚀 AppInsightsHandler.on_llm_start called!")
            
            # Extract useful information
            model_name = serialized.get("name", "unknown")
            model_type = serialized.get("_type", "unknown")
            
            custom_dimensions = self._get_custom_dimensions(
                event_type="llm_start",
                model_name=model_name,
                model_type=model_type,
                prompt_count=len(prompts),
                prompts=self._safe_serialize(prompts[:2]),  # Log first 2 prompts to avoid too much data
                serialized_keys=list(serialized.keys())
            )
            
            # Add run_id if available
            run_id = kwargs.get("run_id")
            if run_id:
                custom_dimensions["run_id"] = str(run_id)
                self.session_data[str(run_id)] = {
                    "start_time": datetime.now(),
                    "model_name": model_name
                }
            
            app_insights_logger.info(
                f"ARB_Chatbot_LLM_Call_Started - Model: {model_name}",
                extra={"customDimensions": custom_dimensions}
            )
            
        except Exception as e:
            print(f"Error in on_llm_start: {e}")
            app_insights_logger.error(
                f"AppInsights callback error in on_llm_start: {str(e)}",
                extra={"customDimensions": {"error": str(e), "traceback": traceback.format_exc()}}
            )

    def on_llm_end(
        self, 
        response: LLMResult, 
        **kwargs: Any
    ) -> None:
        """Called when LLM ends running"""
        try:
            print("✅ AppInsightsHandler.on_llm_end called!")
            
            # Calculate duration if run_id is available
            run_id = kwargs.get("run_id")
            duration = None
            model_name = "unknown"
            
            if run_id and str(run_id) in self.session_data:
                session_info = self.session_data[str(run_id)]
                duration = (datetime.now() - session_info["start_time"]).total_seconds()
                model_name = session_info.get("model_name", "unknown")
                # Clean up session data
                del self.session_data[str(run_id)]
            
            # Extract response information
            total_tokens = 0
            prompt_tokens = 0
            completion_tokens = 0
            
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                total_tokens = token_usage.get('total_tokens', 0)
                prompt_tokens = token_usage.get('prompt_tokens', 0)
                completion_tokens = token_usage.get('completion_tokens', 0)
            
            # Get response text (first generation)
            response_text = ""
            if response.generations and len(response.generations) > 0:
                if len(response.generations[0]) > 0:
                    response_text = response.generations[0][0].text[:500]  # Limit to 500 chars
            
            custom_dimensions = self._get_custom_dimensions(
                event_type="llm_end",
                model_name=model_name,
                duration_seconds=duration,
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                generation_count=len(response.generations) if response.generations else 0,
                response_preview=response_text,
                llm_output_keys=list(response.llm_output.keys()) if response.llm_output else []
            )
            
            if run_id:
                custom_dimensions["run_id"] = str(run_id)
            
            app_insights_logger.info(
                f"ARB_Chatbot_LLM_Call_Completed - Model: {model_name}, Tokens: {total_tokens}",
                extra={"customDimensions": custom_dimensions}
            )
            
        except Exception as e:
            print(f"Error in on_llm_end: {e}")
            app_insights_logger.error(
                f"AppInsights callback error in on_llm_end: {str(e)}",
                extra={"customDimensions": {"error": str(e), "traceback": traceback.format_exc()}}
            )

    def on_llm_error(
        self, 
        error: Union[Exception, KeyboardInterrupt], 
        **kwargs: Any
    ) -> None:
        """Called when LLM errors"""
        try:
            print(f"❌ AppInsightsHandler.on_llm_error called: {error}")
            
            run_id = kwargs.get("run_id")
            model_name = "unknown"
            
            if run_id and str(run_id) in self.session_data:
                model_name = self.session_data[str(run_id)].get("model_name", "unknown")
                # Clean up session data
                del self.session_data[str(run_id)]
            
            custom_dimensions = self._get_custom_dimensions(
                event_type="llm_error",
                model_name=model_name,
                error_type=type(error).__name__,
                error_message=str(error),
                traceback=traceback.format_exc()
            )
            
            if run_id:
                custom_dimensions["run_id"] = str(run_id)
            
            app_insights_logger.error(
                f"ARB_Chatbot_LLM_Call_Error - Model: {model_name}, Error: {str(error)}",
                extra={"customDimensions": custom_dimensions}
            )
            
        except Exception as e:
            print(f"Error in on_llm_error: {e}")
            app_insights_logger.error(
                f"AppInsights callback error in on_llm_error: {str(e)}",
                extra={"customDimensions": {"error": str(e), "traceback": traceback.format_exc()}}
            )

    def on_chain_start(
        self, 
        serialized: Dict[str, Any], 
        inputs: Dict[str, Any], 
        **kwargs: Any
    ) -> None:
        """Called when chain starts running"""
        try:
            print("🔗 AppInsightsHandler.on_chain_start called!")
            
            chain_name = serialized.get("name", "unknown")
            chain_type = serialized.get("_type", "unknown")
            
            custom_dimensions = self._get_custom_dimensions(
                event_type="chain_start",
                chain_name=chain_name,
                chain_type=chain_type,
                input_keys=list(inputs.keys()) if inputs else []
            )
            
            run_id = kwargs.get("run_id")
            if run_id:
                custom_dimensions["run_id"] = str(run_id)
            
            app_insights_logger.info(
                f"ARB_Chatbot_Chain_Started - Chain: {chain_name}",
                extra={"customDimensions": custom_dimensions}
            )
            
        except Exception as e:
            print(f"Error in on_chain_start: {e}")

    def on_chain_end(
        self, 
        outputs: Dict[str, Any], 
        **kwargs: Any
    ) -> None:
        """Called when chain ends running"""
        try:
            print("✅ AppInsightsHandler.on_chain_end called!")
            
            custom_dimensions = self._get_custom_dimensions(
                event_type="chain_end",
                output_keys=list(outputs.keys()) if outputs else []
            )
            
            run_id = kwargs.get("run_id")
            if run_id:
                custom_dimensions["run_id"] = str(run_id)
            
            app_insights_logger.info(
                f"ARB_Chatbot_Chain_Completed",
                extra={"customDimensions": custom_dimensions}
            )
            
        except Exception as e:
            print(f"Error in on_chain_end: {e}")

    def on_tool_start(
        self, 
        serialized: Dict[str, Any], 
        input_str: str, 
        **kwargs: Any
    ) -> None:
        """Called when tool starts running"""
        try:
            print("🔧 AppInsightsHandler.on_tool_start called!")
            
            tool_name = serialized.get("name", "unknown")
            
            custom_dimensions = self._get_custom_dimensions(
                event_type="tool_start",
                tool_name=tool_name,
                input_preview=input_str[:200] if input_str else ""
            )
            
            run_id = kwargs.get("run_id")
            if run_id:
                custom_dimensions["run_id"] = str(run_id)
            
            app_insights_logger.info(
                f"ARB_Chatbot_Tool_Started - Tool: {tool_name}",
                extra={"customDimensions": custom_dimensions}
            )
            
        except Exception as e:
            print(f"Error in on_tool_start: {e}")

    def on_tool_end(
        self, 
        output: str, 
        **kwargs: Any
    ) -> None:
        """Called when tool ends running"""
        try:
            print("✅ AppInsightsHandler.on_tool_end called!")
            
            custom_dimensions = self._get_custom_dimensions(
                event_type="tool_end",
                output_preview=output[:200] if output else ""
            )
            
            run_id = kwargs.get("run_id")
            if run_id:
                custom_dimensions["run_id"] = str(run_id)
            
            app_insights_logger.info(
                f"ARB_Chatbot_Tool_Completed",
                extra={"customDimensions": custom_dimensions}
            )
            
        except Exception as e:
            print(f"Error in on_tool_end: {e}")

    def on_agent_action(self, action, **kwargs: Any) -> None:
        """Called when agent takes an action"""
        try:
            print("🤖 AppInsightsHandler.on_agent_action called!")
            
            custom_dimensions = self._get_custom_dimensions(
                event_type="agent_action",
                tool=action.tool,
                tool_input=self._safe_serialize(action.tool_input)[:200]
            )
            
            run_id = kwargs.get("run_id")
            if run_id:
                custom_dimensions["run_id"] = str(run_id)
            
            app_insights_logger.info(
                f"ARB_Chatbot_Agent_Action - Tool: {action.tool}",
                extra={"customDimensions": custom_dimensions}
            )
            
        except Exception as e:
            print(f"Error in on_agent_action: {e}")

    def on_agent_finish(self, finish, **kwargs: Any) -> None:
        """Called when agent finishes"""
        try:
            print("🏁 AppInsightsHandler.on_agent_finish called!")
            
            custom_dimensions = self._get_custom_dimensions(
                event_type="agent_finish",
                return_values_keys=list(finish.return_values.keys()) if finish.return_values else []
            )
            
            run_id = kwargs.get("run_id")
            if run_id:
                custom_dimensions["run_id"] = str(run_id)
            
            app_insights_logger.info(
                f"ARB_Chatbot_Agent_Finished",
                extra={"customDimensions": custom_dimensions}
            )
            
        except Exception as e:
            print(f"Error in on_agent_finish: {e}")