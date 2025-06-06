import os
from enum import Enum
from typing import Any, Optional, Dict

from langchain_core.language_models.chat_models import BaseChatModel # Assuming this might be used elsewhere or by Langchain internally
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report"""

class SearchAPI(Enum):
    """Enumeration of supported search APIs."""
    TAVILY = "tavily"
    # Other search APIs like PERPLEXITY, EXA, ARXIV, etc., have been removed
    # as per the requirement to use only Tavily.

class Configuration(BaseModel):
    """The configurable fields for the chatbot or report generation system."""

    # Common configuration
    report_structure: str = DEFAULT_REPORT_STRUCTURE # Defaults to the default report structure
    
    # Graph-specific configuration (likely for a research/report generation graph)
    number_of_queries: int = 2 # Number of search queries to generate per iteration
    max_search_depth: int = 2 # Maximum number of reflection + search iterations
    
    # Planner model configuration (for planning the research/report)
    planner_provider: str = "openai"  # Changed to OpenAI
    planner_model: str = "gpt-4.1" # Changed to an OpenAI model (e.g., gpt-4.1, gpt-4o)
    planner_model_kwargs: Optional[Dict[str, Any]] = None # kwargs for planner_model
    
    # Writer model configuration (for writing the report content)
    writer_provider: str = "openai" # Changed to OpenAI
    writer_model: str = "gpt-4.1" # Changed to an OpenAI model (e.g., gpt-4.1, gpt-4o)
    writer_model_kwargs: Optional[Dict[str, Any]] = None # kwargs for writer_model
    
    # Search API configuration
    search_api: SearchAPI = SearchAPI.TAVILY # Default to TAVILY (now the only option)
    search_api_config: Optional[Dict[str, Any]] = None # API-specific configuration (e.g., Tavily API key if not in env)
    
    # Multi-agent specific configuration (if using a multi-agent setup)
    supervisor_model: str = "openai:gpt-4.1" # Model for supervisor agent, already OpenAI
    researcher_model: str = "openai:gpt-4.1" # Model for research agents, already OpenAI

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """
        Create a Configuration instance from a Langchain RunnableConfig.
        
        This method allows for dynamic configuration loading, prioritizing 
        environment variables, then values from the RunnableConfig.
        """
        # Extract 'configurable' dictionary from the RunnableConfig, or use an empty dict if not present
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        
        # Prepare values for Configuration fields
        # It iterates through all fields defined in this Pydantic model
        values: dict[str, Any] = {}
        for field_name in cls.model_fields.keys():
            # Prioritize environment variables (e.g., PLANNER_MODEL from .env)
            # Fallback to value from RunnableConfig's 'configurable' dictionary
            # Fallback to Pydantic field's default value if not found in either
            env_value = os.environ.get(field_name.upper())
            config_value = configurable.get(field_name)
            
            if env_value is not None:
                values[field_name] = env_value
            elif config_value is not None:
                values[field_name] = config_value
            # If neither env_var nor config_value is found, Pydantic will use the default
            # or raise an error if the field is required and has no default.
            # We only add to 'values' if we found something, to allow Pydantic defaults to apply.

        # Create and return a Configuration instance
        # Only pass keys that were explicitly found in environment or config,
        # or that have defaults defined in the model.
        # Pydantic handles parsing and type conversion.
        # The original code's **{k: v for k, v in values.items() if v is not None}
        # was slightly problematic if a config explicitly set a value to None
        # when the default was something else.
        # A better approach is to let Pydantic handle defaults by not passing the key if no override exists.
        
        # Corrected logic for value prioritization:
        final_values_for_instantiation: dict[str, Any] = {}
        for field_name in cls.model_fields.keys():
            # Check environment variable first
            env_var_value = os.environ.get(field_name.upper())
            if env_var_value is not None:
                final_values_for_instantiation[field_name] = env_var_value
                continue # Found in env, use this

            # Check RunnableConfig next
            if field_name in configurable: # Check if key exists, even if value is None
                final_values_for_instantiation[field_name] = configurable[field_name]
                continue # Found in config, use this
            
            # If not in env or config, Pydantic will use the field's default value
            # when the instance is created. So, no need to add it to final_values_for_instantiation here.

        return cls(**final_values_for_instantiation)

