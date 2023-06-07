from langchain.llms import OpenAI
from langchain.tools.json.tool import JsonSpec
from langchain.agents import create_json_agent, AgentExecutor
import json
from langchain.agents.agent_toolkits import JsonToolkit
from utils.helper_functions import *

setup_openAI()

with open("data/example_data.json", "r") as f:
    data = json.load(f)

json_spec = JsonSpec(dict_=data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm=OpenAI(temperature=0), toolkit=json_toolkit, verbose=True
)
while True:
    print("type your question")
    json_obj = json_agent_executor.run(input())
    #print(json_obj["Thought"][-1])

