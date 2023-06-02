from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import os
def setup_openAI():
    os.environ["OPENAI_API_KEY"] = "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"

def preprocess_texts(raw_text):
    '''
    @param raw_text: the concatinated text to be processed
    @return texts: the splitted and tokenized text
    '''
    text_splitter = CharacterTextSplitter(
                        separator = "\n",
                        chunk_size = 1024,
                        chunk_overlap  = 200,
                        length_function = len,
                    )
    texts = text_splitter.split_text(raw_text)
    return texts

def read_pdf_text(path, preprocess_langchain=False):
    '''
    @param path: the pdf object path
    @param preprocess_langchain: preprocessing flag from langchain
    @return texts: all the text from the pdf concatinated
    '''
    reader = PdfReader(path)
    raw_text = ''

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    if preprocess_langchain:
        texts = preprocess_texts(raw_text)
    else:
        texts = raw_text
    return texts

def process_all_pdfs(directory_path, preprocess_langchain=False):
    '''
    @param directory_path: get the directory of the documentstore
    @param preprocess_langchain: if the preprocess for langchain to optimize token in chunks should be done
    @param returns: all the concatinated texts from pdfs
    '''
    all_texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory_path, filename)
            texts = read_pdf_text(filepath, preprocess_langchain)
            all_texts.extend(texts)
    return all_texts
