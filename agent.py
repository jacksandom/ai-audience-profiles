from typing import Any, Generator, Optional, Sequence, Union

import json
import os
import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langchain.prompts import PromptTemplate
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

mlflow.langchain.autolog()

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

system_prompt = PromptTemplate(
    input_variables=["segment", "profile"],
    template="""
    You are an audience persona named {segment} with the following profile:
    {profile}

    The user is an advertising content writer and wants to tailor copy specific to your persona. Your goal is to assist the user in doing this by acting as a {segment} and helping the user to test ideas.

    If asked to improve a specific piece of ad conent, provide 3-5 actionable concise recommendations to make this ad more appealing to the customer profile. Give an example of an improved ad text.
    Your response should follow this structured format:
    
    - **Highlight Key Features:** (What should be emphasised?)
    - **Tone Adjustments:** (How should the messaging be modified?)
    - **Messaging Strategies:** (What persuasive elements should be included?)

    **Improved ad text**

    Do not do this unless asked to do so.

    Stay in character always and respond to questions as this persona. Only respond in the context of your audience persona but don't refer to yourself by the segment name. If asked about something unrelated, politely redirect the conversation.
    """
)

tools = []

JSON_PATH = "model_artifacts/profiles.json"  # Local path before MLflow logging

#####################
## Define agent logic
#####################

def get_customer_profile(custom_inputs, profiles):
    """
    Retrieves a predefined customer profile based on the segment.
    If provided segment is invalid, chooses a default profile.
    """
    segment = custom_inputs.get("segment", "Casual Users")
    return profiles.get(segment, "No profile available.")


def create_profile_agent(
    model: LanguageModelLike,
    agent_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        return "continue" if last_message.get("tool_calls") else "end"
    
    def generate_prompt_with_profile(state: ChatAgentState):
        """
        Retrieves the customer profile and formats the system prompt dynamically.
        """
        custom_inputs = state.get("custom_inputs", {})
        profile = get_customer_profile(custom_inputs, state["context"].get("profiles", {}))

        formatted_prompt = system_prompt.format(
            segment=custom_inputs.get("segment", "Casual Users"),
            profile=profile
        )

        # Store the profile in context so it persists during the chat
        state["context"]["customer_profile"] = profile

        return [{"role": "system", "content": formatted_prompt}] + state["messages"]

    preprocessor = RunnableLambda(generate_prompt_with_profile)
    model_runnable = preprocessor | model


    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}


    workflow = StateGraph(ChatAgentState)
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.set_entry_point("agent")

    return workflow.compile()


class LangGraphChatAgent(ChatAgent, mlflow.pyfunc.PythonModel):
    def __init__(self, agent: CompiledStateGraph, profiles_path: str = None):
        self.agent = agent
        self.PROFILES = {}

        # Load profiles locally if available (before logging the model)
        if profiles_path and os.path.exists(profiles_path):
            print(f"✅ Loading profiles from local JSON at {profiles_path}")
            with open(profiles_path, "r") as f:
                self.PROFILES = json.load(f)
        else:
            print(f"profiles.json not found locally. Will load from context.")

    def load_context(self, context):
        """
        Loads customer profiles from MLflow artifacts when the model is served.
        """
        config_path = context.artifacts.get("config")
        json_path = os.path.join(config_path, "profiles.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"profiles.json not found at {json_path}")

        with open(json_path, "r") as f:
            self.PROFILES = json.load(f)

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """
        Uses the loaded profiles.json to generate responses.
        """
        custom_inputs = custom_inputs or {}
        segment = custom_inputs.get("segment", "Casual Users")
        profile = self.PROFILES.get(segment, "No profile available.")

        request = {
            "messages": self._convert_messages_to_dict(messages),
            "custom_inputs": custom_inputs,
            "context": context.model_dump_compat() if context else {},
        }

        response = ChatAgentResponse(messages=[])
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                if not node_data:
                    continue
                for msg in node_data.get("messages", []):
                    response.messages.append(ChatAgentMessage(**msg))
                if "custom_outputs" in node_data:
                    response.custom_outputs = node_data["custom_outputs"]
        return response
    
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """
        Uses the loaded profiles.json to generate responses.
        """
        custom_inputs = custom_inputs or {}
        segment = custom_inputs.get("segment", "Casual Users")
        profile = self.PROFILES.get(segment, "No profile available.")

        request = {
            "messages": self._convert_messages_to_dict(messages),
            "custom_inputs": custom_inputs,
            "context": context.model_dump_compat() if context else {},
        }

        response = ChatAgentResponse(messages=[])
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                if not node_data:
                    continue
                messages = node_data.get("messages", [])
                custom_outputs = node_data.get("custom_outputs")
                for i, message in enumerate(messages):
                    chunk = {"delta": message}
                    # Only emit custom_outputs with the last streaming chunk from this node
                    if custom_outputs and i == len(messages) - 1:
                        chunk["custom_outputs"] = custom_outputs
                    yield ChatAgentChunk(**chunk)


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
agent = create_profile_agent(llm, system_prompt)
AGENT = LangGraphChatAgent(agent, JSON_PATH)
mlflow.models.set_model(AGENT)
