from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import VectorDBQA, RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from utils.helper_functions import *

# Create a completion
setup_openAI()
llm = OpenAI()

# initialize the embeddings using openAI ada text embedding library
embeddings = OpenAIEmbeddings()
# embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key, chunk_size=1)

# initialize and read the *.pdf object
texts = process_all_pdfs("pdf_data", preprocess_langchain=True)

# initialize the FAISS document store using the preprocessed text and initialized embeddings
docsearch = FAISS.from_texts(texts, embeddings)
retriever = docsearch.as_retriever()
# Create a conversation buffer memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0), retriever=retriever, memory=memory
)

chat_history = []
while True:
    # define the question
    print("type your question")
    query = input("")
    result = qa({"question": query, "chat_history": chat_history})
    print("system: ", result["answer"])
    chat_history.append((query, result["answer"]))

