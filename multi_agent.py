from typing import Literal, Dict, List, Union 

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send # type: ignore
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command # type: ignore # interrupt will be bypassed

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Section, 
    Queries,
    SearchQuery, 
    Feedback
)

from prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)

from configuration import Configuration
from  tools1 import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search
)

# --- Graph Node Definitions ---

async def generate_report_plan(state: ReportState, config: RunnableConfig) -> Dict[str, List[Section]]:
    """
    Generate the initial report plan with sections.
    """
    topic = state["topic"]
    feedback_list = state.get("feedback_on_report_plan", [])
    feedback = " /// ".join(feedback_list) if feedback_list else "" 

    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}
    params_to_pass = get_search_params(search_api, search_api_config)

    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    researcher_model_name = get_config_value(configurable.researcher_model)
    provider_for_researcher = get_config_value(configurable.writer_provider) 
    researcher_model_kwargs = {} 
    
    initial_query_writing_llm = init_chat_model(
        model=researcher_model_name, 
        model_provider=provider_for_researcher, 
        model_kwargs=researcher_model_kwargs
    ).with_structured_output(Queries)

    query_writer_prompt = report_planner_query_writer_instructions.format(
        topic=topic, 
        report_organization=report_structure, 
        number_of_queries=number_of_queries
    )
    query_generation_result = await initial_query_writing_llm.ainvoke([
        SystemMessage(content=query_writer_prompt),
        HumanMessage(content="Generate search queries that will help with planning the sections of the report.")
    ])
    
    query_list_str = [sq.search_query for sq in query_generation_result.queries]
    source_str = await select_and_execute_search(search_api, query_list_str, params_to_pass)

    planner_prompt_system = report_planner_instructions.format(
        topic=topic, 
        report_organization=report_structure, 
        context=source_str, 
        feedback=feedback
    )
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model_name = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})
    
    planner_llm_instance = init_chat_model(
        model=planner_model_name, 
        model_provider=planner_provider, 
        model_kwargs=planner_model_kwargs
    )
    if planner_model_name == "gpt-4o":
        planner_llm_instance = init_chat_model(
            model=planner_model_name, 
            model_provider=planner_provider, 
            max_tokens=20_000, 
            thinking={"type": "enabled", "budget_tokens": 16_000} # type: ignore
        )
    structured_planner_llm = planner_llm_instance.with_structured_output(Sections)
    
    planner_message_user = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
Each section must have: name, description, research, and content fields."""

    report_sections_result = await structured_planner_llm.ainvoke([
        SystemMessage(content=planner_prompt_system),
        HumanMessage(content=planner_message_user)
    ])

    return {"sections": report_sections_result.sections}


def human_feedback_node(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan", "build_section_with_web_research", "gather_completed_sections"]]: # type: ignore
    """
    Bypasses human feedback and automatically approves the report plan for diagnostic purposes.
    """
    print("--- human_feedback_node: Auto-approving plan for diagnostics ---")
    topic = state["topic"]
    sections = state['sections']
    
    # Simulate automatic approval
    feedback_input = True 

    if isinstance(feedback_input, bool) and feedback_input is True:
        research_tasks = [
            Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
            for s in sections 
            if s.research
        ]
        if research_tasks:
            print(f"--- human_feedback_node: Sending {len(research_tasks)} sections to 'build_section_with_web_research' ---")
            return Command(goto=research_tasks)
        else:
            print("--- human_feedback_node: No research sections. Proceeding to 'gather_completed_sections' ---")
            return Command(goto="gather_completed_sections")
    
    # This part should ideally not be reached with auto-approval
    elif isinstance(feedback_input, str) and feedback_input.strip():
        print("--- human_feedback_node: (This should not happen with auto-approval) Regenerating plan due to simulated feedback ---")
        return Command(goto="generate_report_plan", update={"feedback_on_report_plan": [feedback_input]})
    else:
        # This case should also not be reached
        raise TypeError(
            f"Interrupt value of type {type(feedback_input)} or empty feedback is not supported. "
            "This node is set to auto-approve."
        )

async def generate_queries_node(state: SectionState, config: RunnableConfig) -> Dict[str, List[SearchQuery]]:
    """
    Generate search queries for researching a specific section (uses researcher_model).
    """
    topic = state["topic"]
    section = state["section"]

    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    researcher_model_name = get_config_value(configurable.researcher_model)
    provider_for_researcher = get_config_value(configurable.writer_provider) 
    researcher_model_kwargs = {} 
    
    query_writing_llm = init_chat_model(
        model=researcher_model_name, 
        model_provider=provider_for_researcher, 
        model_kwargs=researcher_model_kwargs
    ).with_structured_output(Queries)

    system_instructions = query_writer_instructions.format(
        topic=topic, 
        section_topic=section.description, 
        number_of_queries=number_of_queries
    )
    queries_result = await query_writing_llm.ainvoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate search queries on the provided topic.")
    ])
    return {"search_queries": queries_result.queries}

async def search_web_node(state: SectionState, config: RunnableConfig) -> Dict[str, any]:
    """
    Execute web searches for the section queries.
    """
    search_queries = state["search_queries"]
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}
    params_to_pass = get_search_params(search_api, search_api_config)

    query_list_str = [query.search_query for query in search_queries]
    print(f"--- search_web_node: Executing search for section '{state['section'].name}', API: {search_api}, Queries: {query_list_str}, Params: {params_to_pass} ---")
    source_str = await select_and_execute_search(search_api, query_list_str, params_to_pass)
    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}

async def write_section_node(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]: # type: ignore
    """
    Write a section (using writer_model) and evaluate (using researcher_model).
    """
    topic = state["topic"]
    section = state["section"] 
    source_str = state["source_str"]
    configurable = Configuration.from_runnable_config(config)
    print(f"--- write_section_node: Writing section '{section.name}' ---")

    section_writer_inputs_formatted = section_writer_inputs.format(
        topic=topic, 
        section_name=section.name, 
        section_topic=section.description, 
        context=source_str, 
        section_content=section.content
    )
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    section_writing_llm = init_chat_model(
        model=writer_model_name, 
        model_provider=writer_provider, 
        model_kwargs=writer_model_kwargs
    )
    section_content_result = await section_writing_llm.ainvoke([
        SystemMessage(content=section_writer_instructions),
        HumanMessage(content=section_writer_inputs_formatted)
    ])
    section.content = section_content_result.content
    print(f"--- write_section_node: Section '{section.name}' content generated. Now grading. ---")

    section_grader_message_user = (
        "Grade the report section and consider follow-up questions for missing information. "
        "If the grade is 'pass', return empty strings for all follow-up queries. "
        "If the grade is 'fail', provide specific search queries to gather missing information."
    )
    section_grader_instructions_formatted = section_grader_instructions.format(
        topic=topic, 
        section_topic=section.description,
        section=section.content, 
        number_of_follow_up_queries=configurable.number_of_queries
    )
    
    researcher_model_name_for_grade = get_config_value(configurable.researcher_model)
    provider_for_researcher_grade = get_config_value(configurable.writer_provider)
    researcher_model_kwargs_for_grade = {}

    reflection_llm_instance = init_chat_model(
        model=researcher_model_name_for_grade, 
        model_provider=provider_for_researcher_grade, 
        model_kwargs=researcher_model_kwargs_for_grade
    )
    if researcher_model_name_for_grade == "gpt-4o":
         reflection_llm_instance = init_chat_model(
            model=researcher_model_name_for_grade, 
            model_provider=provider_for_researcher_grade, 
            max_tokens=16000, 
            thinking={"type": "enabled", "budget_tokens": 16_000} # type: ignore
        )
    structured_reflection_llm = reflection_llm_instance.with_structured_output(Feedback)

    feedback_result = await structured_reflection_llm.ainvoke([
        SystemMessage(content=section_grader_instructions_formatted),
        HumanMessage(content=section_grader_message_user)
    ])
    print(f"--- write_section_node: Section '{section.name}' graded. Grade: {feedback_result.grade}, Iteration: {state['search_iterations']} ---")

    if feedback_result.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        print(f"--- write_section_node: Section '{section.name}' PASSED or max depth reached. Ending sub-graph. ---")
        return Command(update={"completed_sections": [section]}, goto=END)
    else:
        print(f"--- write_section_node: Section '{section.name}' FAILED. Going back to 'search_web' for more research. ---")
        return Command(update={"search_queries": feedback_result.follow_up_queries, "section": section}, goto="search_web")

async def write_final_sections_node(state: SectionState, config: RunnableConfig) -> Dict[str, List[Section]]:
    """
    Write sections that don't require research (uses writer_model).
    """
    configurable = Configuration.from_runnable_config(config)
    topic = state["topic"]
    section = state["section"] 
    completed_report_sections_context = state.get("report_sections_from_research", "")
    print(f"--- write_final_sections_node: Writing non-research section '{section.name}' ---")

    system_instructions = final_section_writer_instructions.format(
        topic=topic, 
        section_name=section.name, 
        section_topic=section.description, 
        context=completed_report_sections_context
    )
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    final_section_writer_llm = init_chat_model(
        model=writer_model_name, 
        model_provider=writer_provider, 
        model_kwargs=writer_model_kwargs
    )
    section_content_result = await final_section_writer_llm.ainvoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate a report section based on the provided topic, description, and context from other sections.")
    ])
    section.content = section_content_result.content
    return {"completed_sections": [section]}

def gather_completed_sections_node(state: ReportState) -> Dict[str, str]:
    """
    Formats completed sections into a single string for context.
    """
    completed_sections = state.get("completed_sections", []) 
    print(f"--- gather_completed_sections_node: Gathering {len(completed_sections)} completed sections. ---")
    report_sections_str = format_sections(completed_sections)
    return {"report_sections_from_research": report_sections_str}

def compile_final_report_node(state: ReportState) -> Dict[str, str]:
    """
    Compile all sections into the final report.
    """
    planned_sections = state["sections"] 
    completed_content_map = {s.name: s.content for s in state.get("completed_sections", [])}
    print(f"--- compile_final_report_node: Compiling final report from {len(planned_sections)} planned sections. ---")
    final_report_parts = []
    for planned_section in planned_sections:
        content = completed_content_map.get(planned_section.name, f"[Content for section '{planned_section.name}' not found/completed]")
        content_str = str(content) if content is not None else f"[Content for section '{planned_section.name}' is None]"
        final_report_parts.append(f"## {planned_section.name}\n\n{content_str}")

    full_report = "\n\n".join(final_report_parts)
    return {"final_report": full_report}

def initiate_final_section_writing_edge(state: ReportState) -> List[Send]:
    """
    Edge function to create parallel tasks for writing non-research sections.
    """
    tasks = [
        Send(
            "write_final_sections", 
            {
                "topic": state["topic"], 
                "section": s, 
                "report_sections_from_research": state["report_sections_from_research"]
            }
        ) 
        for s in state["sections"] 
        if not s.research
    ]
    if tasks:
        print(f"--- initiate_final_section_writing_edge: Sending {len(tasks)} non-research sections to 'write_final_sections'. ---")
    return tasks

# --- Graph Construction ---

section_builder_graph = StateGraph(SectionState, output=SectionOutputState) # type: ignore
section_builder_graph.add_node("generate_queries", generate_queries_node)
section_builder_graph.add_node("search_web", search_web_node)
section_builder_graph.add_node("write_section", write_section_node)
section_builder_graph.add_edge(START, "generate_queries")
section_builder_graph.add_edge("generate_queries", "search_web")
section_builder_graph.add_edge("search_web", "write_section")

report_builder_graph = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration) # type: ignore
report_builder_graph.add_node("generate_report_plan", generate_report_plan)
report_builder_graph.add_node("human_feedback", human_feedback_node) # Now auto-approves
report_builder_graph.add_node("build_section_with_web_research", section_builder_graph.compile())
report_builder_graph.add_node("gather_completed_sections", gather_completed_sections_node)
report_builder_graph.add_node("write_final_sections", write_final_sections_node) 
report_builder_graph.add_node("compile_final_report", compile_final_report_node)

report_builder_graph.add_edge(START, "generate_report_plan")
report_builder_graph.add_edge("generate_report_plan", "human_feedback")
report_builder_graph.add_edge("build_section_with_web_research", "gather_completed_sections")

def router_after_gather_sections(state: ReportState) -> Union[List[Send], str]:
    """
    Router function to decide if non-research sections need to be written.
    If so, returns Send commands generated by initiate_final_section_writing_edge.
    Otherwise, returns the name of the next node ("compile_final_report").
    """
    if any(not s.research for s in state["sections"]):
        print("--- router_after_gather_sections: Non-research sections found. Initiating tasks. ---")
        return initiate_final_section_writing_edge(state) 
    else:
        print("--- router_after_gather_sections: No non-research sections. Proceeding to compile report. ---")
        return "compile_final_report"

report_builder_graph.add_conditional_edges(
    "gather_completed_sections",
    router_after_gather_sections,
    {
        "compile_final_report": "compile_final_report"
        # If router_after_gather_sections returns List[Send], LangGraph handles dispatching.
        # The next node after those Sends complete will be determined by the edge from 'write_final_sections'.
    }
)

report_builder_graph.add_edge("write_final_sections", "compile_final_report")
report_builder_graph.add_edge("compile_final_report", END)

graph = report_builder_graph.compile()

