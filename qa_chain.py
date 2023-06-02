from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import VectorDBQA,  RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from utils.helper_functions import *

# Create a completion
setup_openAI()
llm = OpenAI()

#initialize the embeddibgs using openAI ada text embedding library
embeddings = OpenAIEmbeddings()
#embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key, chunk_size=1)

#initialize and read the *.pdf object
texts = process_all_pdfs('data/pdf_data', preprocess_langchain=True)
#reader = PdfReader('data/pdf_data/nutrition_ short_en-GB.pdf')
#read the pdf texts
# texts = read_pdf_text(reader, preprocess_langchain=True)
#extract and preprocess the text
#texts = preprocess_texts(all_text)

#initialize the FAISS document store using the preprocessed text and initialized embeddings
docsearch = FAISS.from_texts(texts, embeddings)
retriever = docsearch.as_retriever()
#ask question using langchain vectordbqa library from the FAISS vector store using using the large language model
#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever=retriever, memory=memory)

#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
chat_history = []
while True:
    #define the question
    print('type your question')
    query = input('')
    result = qa({"question": query, "chat_history": chat_history})
    print('system: ', result["answer"])
    chat_history.append((query, result["answer"]))
    #print(chat_history)