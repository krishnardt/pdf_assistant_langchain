from langchain_text_splitters import TokenTextSplitter
import os, time
from langchain_community.document_loaders import PyPDFLoader

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


os.environ['OPENAI_API_KEY'] = ""



filepath = ("pdf_documents/Indian Polity - 5th Edition - M Laxmikanth.pdf")

chunk_size = 1000
chunk_overlap = chunk_size//10

CHROMA_PATH = "chroma"




lstart = time.time()
loader = PyPDFLoader(filepath)




text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)


pages = loader.load()
# pages = loader.load_and_split(text_splitter)


texts = text_splitter.split_text(pages)
print(texts[0])
