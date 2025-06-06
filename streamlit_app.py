import streamlit as st
import asyncio
import os

# Make sure other project files are accessible
# e.g., by being in the same directory or in PYTHONPATH
from multi_agent import graph as multi_agent_graph # Using the graph from multi_agent_py_v3
from configuration import Configuration
from state import ReportStateInput # To define the input schema for the graph
from langchain_core.runnables import RunnableConfig

# Helper function to run the graph (adapted from run_research.py)
async def run_graph_for_streamlit(topic: str) -> str | None:
    """
    Runs the multi-agent graph for Streamlit and retrieves the final report.
    """
    st.write(f"Starting research for topic: {topic}...")
    st.info("The graph is now running. Please check the console for detailed print logs from the agent's execution steps. This might take a few minutes.")

    initial_input: ReportStateInput = {"topic": topic}

    # Configuration overrides for this specific run.
    # Ensure API keys (OPENAI_API_KEY, TAVILY_API_KEY) are set in your environment.
    try:
        config_override = Configuration(
            researcher_model="gpt-4.1", 
            planner_model="gpt-4.1",    
            writer_model="gpt-4.1",     
            search_api="tavily",        
            max_search_depth=1, # Keep this low for faster Streamlit interaction initially         
            number_of_queries=2         
        )
        runnable_config = RunnableConfig(configurable=config_override.model_dump())
    except Exception as e:
        st.error(f"Error initializing configuration: {e}")
        return None
    
    final_output_state = None
    try:
        # Invoke the graph
        # The print statements from multi_agent_py_v3 will show up in the console
        final_output_state = await multi_agent_graph.ainvoke(initial_input, config=runnable_config)

        if final_output_state and isinstance(final_output_state, dict):
            if 'final_report' in final_output_state and final_output_state['final_report']:
                st.success("Report generated successfully!")
                return final_output_state['final_report']
            else:
                st.warning("Report generation finished, but the final report is empty or missing.")
                st.json(final_output_state) # Display the final state for debugging
                return None
        else:
            st.error("Graph execution did not return a valid final state.")
            st.write("Final output from graph:", final_output_state)
            return None
            
    except Exception as e:
        st.error(f"An error occurred during agent execution: {str(e)}")
        st.exception(e) # Displays the full traceback in Streamlit
        return None

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="AI Research Report Generator")

st.title("🤖 AI Research Report Generator")
st.markdown("""
This application uses a multi-agent system  to generate a research report
on a given topic.
""")

st.sidebar.header("Configuration Notes")
st.sidebar.info("""
- Ensure `OPENAI_API_KEY` and `TAVILY_API_KEY` are set as environment variables.
- The graph uses `gpt-4.1 for Planning, Individual Section writing, and Final Report Writing.
- The 'Human Feedback' step in the graph is currently auto-approved for this demo.
- Check the console output in Langsmith for detailed logs of the agent's execution steps.
""")

# Initialize session state variables
if "report" not in st.session_state:
    st.session_state.report = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

# Input for the research topic
topic_input = st.text_input("Enter the research topic:", placeholder="e.g., The future of AI in healthcare diagnostics")

if st.button("Generate Report", type="primary", disabled=st.session_state.is_loading):
    if topic_input:
        st.session_state.is_loading = True
        st.session_state.report = None # Clear previous report
        st.session_state.error_message = None # Clear previous error

        # Use st.spinner for a visual loading indicator
        with st.spinner("Generating report... This may take several minutes. Please wait."):
            # Run the asynchronous graph execution
            # asyncio.run() is suitable here as Streamlit callbacks are synchronous
            try:
                report_content = asyncio.run(run_graph_for_streamlit(topic_input))
                if report_content:
                    st.session_state.report = report_content
                else:
                    # Error messages are handled within run_graph_for_streamlit and displayed using st.error/warning
                    # If it returns None without specific st messages, we add a generic one.
                    if not st.session_state.error_message: # Check if an error was already logged by the function
                         st.session_state.error_message = "Failed to generate report. Check logs for details."
            except Exception as e:
                # Catch any other unexpected errors from asyncio.run or the function itself
                st.session_state.error_message = f"An unexpected error occurred: {str(e)}"
                st.exception(e)
        
        st.session_state.is_loading = False
        st.rerun() # Rerun to update UI based on new session state
    else:
        st.warning("Please enter a research topic.")

# Display the report or error message
if st.session_state.is_loading:
    # This message will be shown briefly before the spinner takes over if rerun happens fast
    st.info("Processing... please wait.") 
elif st.session_state.report:
    st.subheader("Generated Report")
    st.markdown(st.session_state.report)
    
    # Add a download button for the report
    st.download_button(
        label="Download Report as Markdown",
        data=st.session_state.report,
        file_name=f"{topic_input.replace(' ', '_')}_report.md" if topic_input else "research_report.md",
        mime="text/markdown",
    )
elif st.session_state.error_message:
    # Errors displayed by run_graph_for_streamlit will appear above this if they use st.error directly.
    # This handles cases where the function might return None and set a general error.
    # To avoid double error messages, ensure run_graph_for_streamlit uses st.error for specific errors.
    # For now, this is a fallback.
    if "error occurred during agent execution" not in st.session_state.error_message and \
       "Graph execution did not return a valid final state" not in st.session_state.error_message and \
       "Report generation finished, but the final report is empty" not in st.session_state.error_message:
        st.error(st.session_state.error_message)


st.markdown("---")
st.markdown("Made for Teleperformance")

