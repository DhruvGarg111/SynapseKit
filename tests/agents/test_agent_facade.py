import pytest

from synapsekit import agent, tool
from synapsekit.llm.base import BaseLLM, LLMConfig


class StubLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig(model="stub", api_key="dummy", provider="stub"))
        self.prompts: list[list[dict]] = []

    async def stream(self, prompt: str, **kw):
        yield prompt

    async def generate_with_messages(self, messages: list[dict], **kw) -> str:
        self.prompts.append(messages)
        user_content = messages[-1]["content"]
        if "Observation: Sunny, 22" in user_content:
            return "Thought: I now know the final answer.\nFinal Answer: Sunny, 22C in Tokyo"
        return "Thought: I should check the weather.\nAction: get_weather\nAction Input: Tokyo"


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"


@pytest.mark.asyncio
async def test_simple_agent_async():
    my_agent = agent(
        model="gpt-4o-mini",
        api_key="dummy",
        tools=[get_weather],
    )

    assert my_agent._executor is not None
    assert my_agent._memory is None
    assert len(my_agent._executor.config.tools) == 1
    assert my_agent._executor.config.tools[0].name == "get_weather"


@pytest.mark.asyncio
async def test_simple_agent_async_runs_decorated_tool(monkeypatch):
    stub_llm = StubLLM()
    monkeypatch.setattr("synapsekit.agents.facade.make_llm", lambda **kwargs: stub_llm)

    my_agent = agent(model="gpt-4o-mini", api_key="dummy", tools=[get_weather])

    answer = await my_agent.arun("What's the weather in Tokyo?")

    assert answer == "Sunny, 22C in Tokyo"
    assert len(stub_llm.prompts) == 2
    assert "Observation: Sunny, 22" in stub_llm.prompts[1][-1]["content"]


def test_simple_agent_sync_updates_memory(monkeypatch):
    class StubExecutor:
        def run_sync(self, prompt: str) -> str:
            return f"answer to {prompt}"

    my_agent = agent(
        model="gpt-4o-mini",
        api_key="dummy",
        tools=[get_weather],
        memory=True,
    )
    monkeypatch.setattr(my_agent, "_executor", StubExecutor())

    answer = my_agent.run("hello")

    assert answer == "answer to hello"
    assert my_agent._memory is not None
    assert my_agent._memory.get_messages() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer to hello"},
    ]


def test_simple_agent_memory_flag():
    my_agent = agent(
        model="gpt-4o-mini",
        api_key="dummy",
        tools=[get_weather],
        memory=True,
    )

    assert my_agent._memory is not None
    assert len(my_agent._memory) == 0


def test_tool_decorator_no_parens():
    @tool
    def multiply(a: int, b: int) -> str:
        """Multiply two numbers."""
        return str(a * b)

    assert multiply.name == "multiply"
    assert multiply.description == "Multiply two numbers."
    assert multiply.parameters["type"] == "object"
    assert "a" in multiply.parameters["properties"]
    assert "b" in multiply.parameters["properties"]
