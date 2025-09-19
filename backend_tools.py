from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
import sqlite3
import requests
from dotenv import load_dotenv

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

def duckduckgo_search(query: str) -> dict:
    return search_tool.run(query)

def get_weather(city: str) -> dict:
    """Fetch current weather using OpenWeatherMap API."""
    api_key = "0a8b1305d351bee28b8c1e3f9610ec7e"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        r = requests.get(url)
        data = r.json()

        if r.status_code != 200:
            return {"error": data.get("message", "Could not fetch weather")}

        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "weather": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }

    except Exception as e:
        return {"error": str(e)}


def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation: add, sub, mul, div."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (AAPL, TSLA, etc)."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price, calculator, get_weather]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State + Graph
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    threads = []
    for checkpoint in checkpointer.list(None):
        raw_thread_id = checkpoint.config["configurable"]["thread_id"]
        thread_id = str(raw_thread_id)   # ✅ ensure string

        state = checkpointer.get({"configurable": {"thread_id": raw_thread_id}})
        messages = state.get("messages", [])

        # Take first user message as title
        user_message = next((msg.content for msg in messages if msg.type == "human"), None)

        if not user_message:
            user_message = "Empty conversation"

        threads.append({
            "thread_id": thread_id,
            "title": user_message[:50] + "..."
        })

    return threads
