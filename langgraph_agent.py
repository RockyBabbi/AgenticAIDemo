"""
Agentic AI System using LangGraph with Gradio UI
This example creates an AI agent that can use tools and make decisions
"""

import os
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import gradio as gr
import httpx

# Define tools that the agent can use


@tool # Decorator tool to tell LLM to use this function as a tool
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo HTML (redirect-safe)."""
    import httpx
    from bs4 import BeautifulSoup

    url = "https://html.duckduckgo.com/html/"  # FIXED URL

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    try:
        with httpx.Client(follow_redirects=True, timeout=10) as client:
            response = client.post(
                url,
                data={"q": query},
                headers=headers
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.select(".result")

        if not results:
            return "DuckDuckGo HTML returned no results."

        lines = ["Web search results:"]

        for i, r in enumerate(results[:5], 1):
            title = r.select_one(".result__a")
            snippet = r.select_one(".result__snippet")

            lines.append(
                f"{i}. {title.text if title else 'No title'}\n"
                f"   {snippet.text if snippet else 'No description'}"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Search failed: {str(e)}"

@tool # Decorator tool to tell LLM to use this function as a tool
def get_weather(location: str) -> str:
    """
    Get current weather for a location using the Open-Meteo free API.
    No API key required.
    """
    try:
        with httpx.Client(timeout=10) as client:
            # Step 1: Geocode location
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"
            geo_response = client.get(geo_url, params={"name": location, "count": 1})
            geo_response.raise_for_status()
            geo_data = geo_response.json()

            if not geo_data.get("results"):
                return f"Could not find location: {location}"

            loc = geo_data["results"][0]
            lat = loc["latitude"]
            lon = loc["longitude"]
            city = loc["name"]
            country = loc.get("country", "")

            # Step 2: Fetch current weather
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_response = client.get(
                weather_url,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": True
                }
            )
            weather_response.raise_for_status()
            weather_data = weather_response.json()

            current = weather_data.get("current_weather", {})
            temperature = current.get("temperature")
            wind_speed = current.get("windspeed")
            weather_code = current.get("weathercode")

            return (
                f"🌤️ **Current weather in {city}, {country}:**\n"
                f"- 🌡️ Temperature: {temperature}°C\n"
                f"- 💨 Wind Speed: {wind_speed} km/h\n"
                f"- 🔢 Weather Code: {weather_code}"
            )

    except Exception as e:
        return f"Weather lookup failed: {str(e)}"


# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Initialize the LLM with tools
tools = [search_web, get_weather]

# Azure OpenAI Configuration
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # e.g., "https://your-resource.openai.azure.com/"
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),  # e.g., "gpt-4o"
    api_version="2024-02-15-preview",  # or latest version
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

# Define the agent node
def agent_node(state: AgentState):
    """The main agent that decides what to do next"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Define whether to continue or end
def should_continue(state: AgentState):
    """Determine if the agent should continue or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are no tool calls, we're done
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    return "continue"

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

# Add edge from tools back to agent
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

# Function to run the agent and return formatted output
def run_agent(user_input: str) -> str:
    """Run the agent with a user input and return the response"""
    if not user_input.strip():
        return "Please enter a question or request."
    
    try:
        output_text = f"**User:** {user_input}\n\n---\n\n"
        
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        step_count = 0
        for output in app.stream(inputs):
            for key, value in output.items():
                step_count += 1
                output_text += f"### Step {step_count}: {key.upper()}\n\n"
                
                if "messages" in value:
                    for msg in value["messages"]:
                        if isinstance(msg, AIMessage):
                            if msg.content:
                                output_text += f"**Assistant:** {msg.content}\n\n"
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    tool_name = tool_call['name']
                                    tool_args = tool_call['args']
                                    output_text += f"🔧 **Tool Call:** `{tool_name}`\n"
                                    output_text += f"   **Arguments:** `{tool_args}`\n\n"
                        elif isinstance(msg, ToolMessage):
                            output_text += f"✅ **Tool Result:** {msg.content}\n\n"
                
                output_text += "---\n\n"
        
        return output_text
    
    except Exception as e:
        return f"❌ **Error:** {str(e)}\n\nPlease check your Azure OpenAI configuration."

# Create Gradio Interface
def create_gradio_interface():
    """Create and launch the Gradio interface"""
    
    # Google-inspired CSS
    custom_css = """
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .logo-section {
        text-align: center;
        padding: 80px 0 40px 0;
    }
    .logo-text {
        font-size: 72px;
        font-weight: 400;
        letter-spacing: -2px;
        margin: 0;
    }
    .logo-ai {
        background: linear-gradient(90deg, #4285f4, #ea4335, #fbbc04, #34a853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .search-box {
        margin: 30px auto;
        max-width: 600px;
    }
    .button-container {
        text-align: center;
        margin: 30px 0;
    }
    .output-section {
        margin-top: 40px;
        padding: 20px;
        background: white;
        border-radius: 8px;
        border: 1px solid #dfe1e5;
    }
    .examples-section {
        text-align: center;
        margin: 40px 0;
    }
    """
    
    with gr.Blocks(
        title="Agentic AI Assistant", 
        theme=gr.themes.Default(
            primary_hue="blue",
            font=gr.themes.GoogleFont("Product Sans")
        ),
        css=custom_css
    ) as demo:
        
        # Google-style Logo
        gr.HTML(
            """
            <div class="logo-section">
                <h1 class="logo-text">
                    <span class="logo-ai">Agentic AI Assistant</span>
                </h1>
            </div>
            """
        )
        
        # Search Box
        with gr.Column(elem_classes="search-box"):
            user_input = gr.Textbox(
                label="",
                placeholder="Ask me anything...",
                lines=1,
                show_label=False,
                container=False
            )
        
        # Button Container
        with gr.Row(elem_classes="button-container"):
            submit_btn = gr.Button("Ask AI", variant="primary", size="lg")
        
        # Examples (Google-style)
        with gr.Column(elem_classes="examples-section"):
            gr.Examples(
                examples=[
                    ["What's the weather in Paris?"],
                    ["Search for Python tutorials"],
                ],
                inputs=user_input,
                label="",
                examples_per_page=3
            )
        
        # Output Section
        with gr.Column(elem_classes="output-section", visible=True) as output_section:
            output = gr.Markdown(
                value="",
                label=""
            )
        
        # Event handlers
        def process_query(query):
            if not query.strip():
                return ""
            return run_agent(query)
        
        submit_btn.click(
            fn=process_query,
            inputs=user_input,
            outputs=output
        )
        
        user_input.submit(
            fn=process_query,
            inputs=user_input,
            outputs=output
        )
    
    return demo

# Main execution
if __name__ == "__main__":
    # Check if Azure OpenAI credentials are set
    required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("⚠️  Warning: Missing Azure OpenAI environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these before running the application.")
    
    # Check if DuckDuckGo is installed
    try:
        from duckduckgo_search import DDGS
        print("✅ DuckDuckGo search is ready!")
    except ImportError:
        print("⚠️  Warning: DuckDuckGo search not installed.")
        print("   Run: pip install duckduckgo-search")
        print("   Web search will not work until installed.\n")
    
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",  # Use localhost
        server_port=7860,  # Default Gradio port
        inbrowser=True  # Automatically open in browser
    )