from langchain.indexes import GraphIndexCreator
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
import os
from utils.helper_functions import *

setup_openAI()

text = "It won’t look like much, but if you stop and look closely, you’ll see a “Field of dreams,” the ground on which America’s future will be built. \nThis is where Intel, the American company that helped build Silicon Valley, is going to build its $20 billion semiconductor “mega site”. \nUp to eight state-of-the-art factories in one place. 10,000 new good-paying jobs."

index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
graph = index_creator.from_text(text)

graph = index_creator.from_text(text)

triples = graph.get_triples()

print(f"detected triples are {triples}")

#connect to wikipedia entries if possible

