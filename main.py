import os
from dotenv import load_dotenv # Loads secrets from .env file
from pydantic import BaseModel # Creates data validation "blueprints"
from langchain_openai import ChatOpenAI # Wrappers to chat with GPT/Claude
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate # Creates reusable conversation templates
from langchain_core.output_parsers import PydanticOutputParser # Translates LLM text into structured data
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_file_tool

# Load environment variables from .env file (api keys)
load_dotenv()


# Step 1: Define what structure you want using Pydantic
# This acts like a form that the LLM must fill out with specific fields (blueprint)
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    source : list[str]
    tools_used: list[str]


# Debug: Check if the API key is loaded
api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"API Key loaded: {api_key[:10] if api_key else 'NOT FOUND'}...")


# Initialize language models
llm = ChatOpenAI(model="gpt-4o-mini")
llm2 = ChatAnthropic(model="claude-3-5-sonnet-20241022")


# Step 2: Create a "translator" that understands this structure (translator)
# converts LLM text responses into structured Python objects
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# creating the prompt that guides ai behavior

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", # sets the character of llm
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools.
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
).partial(format_instructions=parser.get_format_instructions()) #help LLM understand HOW to structure response


# creating simple agent


# Create the AI agent with access to tools
tools = [search_tool, wiki_tool, save_file_tool]
agent = create_tool_calling_agent(

    llm = llm, # The brain that decides what to do
    prompt = prompt, # The instructions that guide the agent
    tools = tools

)

# Create the agent executor that runs the agent and manages tool usage, verbose=true means shows thinking
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose= True)

#gets user input and run research
query = input("What can I help you research today?\n")
# invoke --> executes the research with the user's query
raw_response = agent_executor.invoke({"query": query}) #key matches what your system expects


# Try to parse the raw response into a structured format
try:
    structured_response = parser.parse(raw_response.get("output")) # ouput is fixed key that LangChain's AgentExecutor uses for final result
    # ↑ This conversion RELIES on Pydantic to ensure data quality, Converts to a Python object that matches ResearchResponse format
    # through the creation of the object, we can have easy access to specific data
    
    # print(f"Topic: {structured_response.topic}")
    # print(f"Summary: {structured_response.summary}")
    # print(f"Sources: {', '.join(structured_response.sources)}")
    # print(f"Tools Used: {', '.join(structured_response.tools_used)}")

    print(structured_response)
except Exception as e:
    print("Error parsing repsonse",e, "Raw Response - ", raw_response)


