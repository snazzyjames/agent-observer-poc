from typing import TypedDict

from langgraph.graph import StateGraph, END
import ollama


class PipelineState(TypedDict):
    task: str
    messages: list
    iteration: int
    system_prompt: str
    reasoning: str


def planner_node(state: PipelineState) -> dict:
    response = ollama.chat(model="qwen2.5-coder:14b", messages=state["messages"])
    response_text = response.message.content
    updated_messages = state["messages"] + [
        {"role": "assistant", "content": response_text}
    ]
    return {
        "messages": updated_messages,
        "iteration": state["iteration"] + 1,
        "reasoning": response_text,
    }


def router(state: PipelineState, max_iterations: int) -> str:
    if state["iteration"] >= max_iterations:
        return "end"
    return "continue"


def build_pipeline(task, max_iterations, observer):
    graph = StateGraph(PipelineState)
    graph.add_node("planner", planner_node)
    graph.add_conditional_edges(
        "planner",
        lambda state: router(state, max_iterations),
        {"continue": "planner", "end": END},
    )
    graph.set_entry_point("planner")
    return graph.compile()
