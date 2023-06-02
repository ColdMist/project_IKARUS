import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.json.tool import JsonSpec
from langchain.agents import (
    create_json_agent,
    AgentExecutor
)
import json
from langchain.agents.agent_toolkits import JsonToolkit

# def preprocess_texts(raw_text):
#     '''
#     @param raw_text: the concatinated text to be processed
#     @return texts: the splitted and tokenized text
#     '''
#     text_splitter = CharacterTextSplitter(
#                         separator = "\n",
#                         chunk_size = 1000,
#                         chunk_overlap  = 200,
#                         length_function = len,
#                     )
#     texts = text_splitter.split_text(raw_text)
#     return texts

# def json_file_loader():
#     # open json file
#     with open('flint.json', 'r') as myfile:
#         data = myfile.read()

os.environ["OPENAI_API_KEY"] = "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"
# with open('data/kb_data/evita_kb_new_recent_update.json', 'r') as myfile:
#     data = myfile.read()
#     print(data)

with open('data/kb_data/evita_kb_new_recent_update.json', 'r') as f:
    data = json.load(f)

# with open('data/kb_data/evita_kb_new_recent_update.json', 'r') as myfile:
#     data = myfile.read()
#     print(data)
json_spec = JsonSpec(dict_=data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm=OpenAI(temperature=0),
    toolkit=json_toolkit,
    verbose=True
)
while True:
    print('type your question')
    json_obj = json_agent_executor.run(input())
    print(json_obj['Thought'][-1])
# exit()
#
# print(data)
# loader = TextLoader('data/kb_data/evita_kb_new_recent_update.json')
#
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=6, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# print(len(docs))
#
#
# embeddings = OpenAIEmbeddings()
# db = FAISS.from_documents(docs, embeddings)
# query = "what is a chatgpt clone"
# docs = db.similarity_search(query)
# # print(docs[0].page_content)
# # exit()
# retriever = db.as_retriever()
# retriever.search_kwargs['distance_metric'] = 'cos'
# retriever.search_kwargs['k'] = 4
#
# qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=False)
#
# # What was the restaurant the group was talking about called?
# while True:
#     query = input("Enter query:")
#     # The Hungry Lobster
#     ans = qa({"query": query})
#     print(f"answer:{ans['result']}")