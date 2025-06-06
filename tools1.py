import os
import asyncio
from typing import List, Optional, Dict, Any, Union, Literal # Keep Union if used by Section or other kept items

from dotenv import load_dotenv
from tavily import AsyncTavilyClient
from langchain_core.tools import tool
from langsmith import traceable

# Assuming 'state.py' exists in the same directory or is otherwise accessible
# and defines the 'Section' class. If 'Section' is not used by any kept function,
# this import and 'format_sections' could also be reviewed.
# For now, 'format_sections' is kept as a general utility.
from state import Section

# Load environment variables from .env file
load_dotenv()

def get_config_value(value: Any) -> Any:
    """
    Helper function to handle string, dict, and enum cases of configuration values.
    Enums are expected to have a 'value' attribute.
    """
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    elif hasattr(value, 'value'): # Check if it's an Enum-like object with a .value
        return value.value
    return value # Fallback for other types

def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filters the search_api_config dictionary to include only parameters accepted by the specified search API.
    Currently, only Tavily is supported.

    Args:
        search_api (str): The search API identifier (should be "tavily").
        search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    # Define accepted parameters for each search API
    # Simplified to only include Tavily
    SEARCH_API_PARAMS = {
        "tavily": ["max_results", "topic", "include_raw_content", "search_depth", "include_answer", "include_images", "include_domains", "exclude_domains"],
    }

    # Get the list of accepted parameters for the given search API
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}

def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str

@traceable
async def tavily_search_async(
    search_queries: List[str], 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance","industry","company","technolgy","product","service","market"] = "general", 
    include_raw_content: bool = True,
    search_depth: Literal["basic", "advanced"] = "basic",
    include_answer: bool = False,
    include_images: bool = False,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Performs concurrent web searches with the Tavily API.

    Args:
        search_queries (List[str]): List of search queries to process.
        max_results (int): Maximum number of results to return.
        topic (Literal["general", "news", "finance"]): Topic to filter results by.
        include_raw_content (bool): Whether to include raw content in the results.
        search_depth (Literal["basic", "advanced"]): The depth of the search.
        include_answer (bool): Whether to include a direct answer to the query.
        include_images (bool): Whether to include images in the results.
        include_domains (Optional[List[str]]): A list of domains to exclusively search for.
        exclude_domains (Optional[List[str]]): A list of domains to exclude from search.


    Returns:
        List[dict]: List of search responses from Tavily API:
            [
                {
                    'query': str,
                    'follow_up_questions': Optional[list], 
                    'answer': Optional[str],
                    'images': Optional[list],
                    'results': [ 
                        {
                            'title': str, 
                            'url': str, 
                            'content': str, 
                            'score': float, 
                            'raw_content': Optional[str] 
                        },
                        ...
                    ]
                },
                ...
            ]
    """
    # Initialize the asynchronous Tavily client
    # It's generally better to initialize the client once if possible,
    # but for a standalone function, this is acceptable.
    # Ensure TAVILY_API_KEY is set in your environment variables.
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []

    for query in search_queries:
        search_tasks.append(
            tavily_async_client.search(
                query=query,
                search_depth=search_depth,
                include_answer=include_answer,
                include_images=include_images,
                max_results=max_results,
                include_raw_content=include_raw_content,
                topic=topic,
                include_domains=include_domains,
                exclude_domains=exclude_domains
            )
        )

    # Execute all searches concurrently
    try:
        search_docs = await asyncio.gather(*search_tasks)
    except Exception as e:
        print(f"An error occurred during Tavily search: {e}")
        # Return a list of error objects, one for each query
        return [
            {
                "query": q, 
                "error": str(e), 
                "results": [], 
                "follow_up_questions": None, 
                "answer": None, 
                "images": None
            } for q in search_queries
        ]
    return search_docs


@tool
async def tavily_search(
    queries: List[str], 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance"] = "general",
    search_depth: Literal["basic", "advanced"] = "basic",
    include_answer: bool = False,
    include_images: bool = False,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None
) -> str:
    """
    Fetches results from Tavily search API and formats them into a string.
    This tool uses tavily_search_async to get content directly, including raw_content.

    Args:
        queries (List[str]): List of search queries.
        max_results (int): Maximum number of results to return per query.
        topic (Literal["general", "news", "finance"]): Topic to filter results by.
        search_depth (Literal["basic", "advanced"]): The depth of the search.
        include_answer (bool): Whether to include a direct answer to the query.
        include_images (bool): Whether to include images in the results.
        include_domains (Optional[List[str]]): A list of domains to exclusively search for.
        exclude_domains (Optional[List[str]]): A list of domains to exclude from search.

    Returns:
        str: A formatted string of search results, deduplicated by URL.
    """
    # Call tavily_search_async, ensuring include_raw_content is True to get full content
    search_responses = await tavily_search_async(
        search_queries=queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True, # Hardcoded to True as this tool formats raw_content
        search_depth=search_depth,
        include_answer=include_answer,
        include_images=include_images,
        include_domains=include_domains,
        exclude_domains=exclude_domains
    )

    formatted_output = "Search results:\n\n"
    
    # Deduplicate results by URL across all query responses
    unique_results_by_url: Dict[str, Dict[str, Any]] = {}
    for response in search_responses:
        if response.get("error"):
            formatted_output += f"Error for query '{response['query']}': {response['error']}\n\n"
            continue
        if response.get("answer"):
             formatted_output += f"Direct Answer for query '{response['query']}': {response['answer']}\n\n"

        for result in response.get('results', []):
            url = result.get('url')
            if url and url not in unique_results_by_url:
                unique_results_by_url[url] = result
    
    if not unique_results_by_url:
        if not any(res.get("answer") for res in search_responses if not res.get("error")): # Check if there was any answer
             return "No valid search results or answers found. Please try different search queries."

    # Format the unique results
    for i, (url, result) in enumerate(unique_results_by_url.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result.get('title', 'N/A')} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result.get('content', 'N/A')}\n\n" # 'content' is the summary from Tavily
        if result.get('raw_content'):
            # Limit raw_content display length for brevity in the final string output
            raw_content_preview = result['raw_content'][:10000] + ("..." if len(result['raw_content']) > 10000 else "")
            formatted_output += f"FULL CONTENT (preview):\n{raw_content_preview}"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    return formatted_output.strip()


async def select_and_execute_search(search_api: str, query_list: list[str], params_to_pass: dict) -> str:
    """
    Select and execute the appropriate search API.
    Currently, only Tavily search is supported.

    Args:
        search_api (str): Name of the search API to use (should be "tavily").
        query_list (list[str]): List of search queries to execute.
        params_to_pass (dict): Parameters to pass to the search API.

    Returns:
        str: Formatted string containing search results.
        
    Raises:
        ValueError: If an unsupported search API is specified.
    """
    print(f"Executing search for API: {search_api}, Queries: {query_list}, Params: {params_to_pass}")
    if search_api == "tavily":
        # The tavily_search tool expects 'queries' as a key in its input dictionary.
        tool_input = {'queries': query_list, **params_to_pass}
        return await tavily_search.ainvoke(tool_input)
    else:
        raise ValueError(f"Unsupported search API: {search_api}. Only 'tavily' is currently supported.")

