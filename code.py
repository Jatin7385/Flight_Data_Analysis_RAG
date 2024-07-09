from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
import os

df = pd.read_csv("../../RAG_Data/RAG_LLM/data/TS_GS_06-22-2024_07_56_11.csv")

# Zero Shot React Description
agent = create_pandas_dataframe_agent(Ollama(model="mistral"),               #OpenAI(temperature=0,openai_api_key=OpenAI_key), 
                                      df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)

print("Data and Model Loaded.....")

while True:
    query = input("Query : ")
    print(agent.invoke(query))