Building an Agentic AI System with LangGraph, Azure OpenAI & Gradio
I’ve been experimenting with agentic AI patterns—moving beyond simple chatbots to systems that can reason, decide, and act. I built a small but complete Agentic AI Assistant using:

⦁	LangGraph for controlled agent workflows
⦁	Azure OpenAI as the reasoning engine
⦁	Tool-based execution (web search & weather)
⦁	Gradio for a clean, interactive UI

🔹 What makes this agent “agentic”?

⦁	The LLM doesn’t just respond—it decides when to use tools
⦁	Tools are executed outside the model, safely and deterministically
⦁	The system loops through think → act → observe → think until completion
⦁	The control flow is explicitly defined using a state graph, not hidden logic

🔹 Key capabilities implemented

⦁	Web search using DuckDuckGo (HTML-safe, redirect-aware)
⦁	Real-time weather lookup via a free Open-Meteo API (no API keys)
⦁	Tool selection and execution orchestrated by LangGraph
⦁	Step-by-step visibility into reasoning and tool usage
⦁	Simple Google-style UI using Gradio


<img width="1193" height="650" alt="image" src="https://github.com/user-attachments/assets/4da49710-a43f-4e8c-a6dc-38c51c8af9b2" />
