import os
from dotenv import load_dotenv # Loads secrets from .env file
from pydantic import BaseModel # Creates data validation "blueprints"
from langchain_openai import ChatOpenAI # Wrappers to chat with GPT/Claude
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate # Creates reusable conversation templates
from langchain_core.output_parsers import PydanticOutputParser # Translates LLM text into structured data
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_file_tool

load_dotenv()


# class to specify type of content that llm generates
class ResearchResponse(BaseModel):
    # blueprint/template - like a form the LLM must fill out
    topic: str
    summary: str
    source : list[str]
    tools_used: list[str]


# Debug: Check if the API key is loaded
api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"API Key loaded: {api_key[:10] if api_key else 'NOT FOUND'}...")


# setup llm
llm = ChatOpenAI(model="gpt-4o-mini")
llm2 = ChatAnthropic(model="claude-3-5-sonnet-20241022")


# Parser = the translator that:
# Tells the LLM "format your answer like ResearchResponse"
# Converts LLM's text response into Python ResearchResponse object

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# creating the prompt

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", # sets the character of llm
            """
            You are a research assistant that will help generate a research paper.
            Answer the use query and use neccessary tools.
            Wrap the output in this format and provide no other text \n{format_instructions}
            """,

        ),
        # Placeholder for chat history (previous messages)
        ("placeholder","{chat_history}"),

        # user question
        ("human","{query}"),

        # Placeholder for agent's thinking process
        ("placeholder", "{agent_scratchpad}"),

    ]
).partial(format_instructions=parser.get_format_instructions())

# creating simple agent

tools = [search_tool, wiki_tool, save_file_tool]
agent = create_tool_calling_agent(

    llm = llm,
    prompt = prompt,
    tools = tools

)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose= True)
query = input("What can I help you research today?\n")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response)
except Exception as e:
    print("Error parsing repsonse",e, "Raw Response - ", raw_response)


