


from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores.chroma import Chroma
# from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

import time

# from langchain_community.embeddings import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import os, shutil

# from chrom_loader import save_to_chroma


os.environ['OPENAI_API_KEY'] = ""



filepath = ("pdf_documents/Indian Polity - 5th Edition - M Laxmikanth.pdf")

chunk_size = 1000
chunk_overlap = chunk_size//10

CHROMA_PATH = "chroma"



'''
lstart = time.time()
loader = PyPDFLoader(filepath)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " "]
)

pages = loader.load()

# pages = loader.load_and_split()

# lend = time.time()

# print(lend-lstart)

# print(pages[0])



chunks = r_splitter.split_documents(pages)

lend = time.time()

print(lend-lstart)


print("Chunks: ", chunks[:5])
print("Length of chunks: ", len(chunks))
print("Pages in the original document: ", len(pages))


print(chunks[50])


def save_to_chroma(chunks: list[Document]):
  """
  Save the given list of Document objects to a Chroma database.
  Args:
  chunks (list[Document]): List of Document objects representing text chunks to save.
  Returns:
  None
  """

  # Clear out the existing database directory if it exists
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

  # Create a new Chroma database from the documents using OpenAI embeddings
  db = Chroma.from_documents(
    chunks,
    OpenAIEmbeddings(),
    persist_directory=CHROMA_PATH
  )

  # Persist the database to disk
  db.persist()
  print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


save_to_chroma(chunks)
'''



query_text = "what is the basic structure of the Indian constitution? Can you explain them in detail?"

# query_text = "How is president elected in India?Explain in detail"


# query_text = "Can you generte 10 multiple choice questions on President of India? Be specific with options. Don't ask question that include years. Ask more conceptual."



# context = "Generate the answer as per Indian constitution in simple and detailed answer"


PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""


embedding_function = OpenAIEmbeddings()

# Prepare the database
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Retrieving the context from the DB using similarity search
results = db.similarity_search_with_relevance_scores(query_text, k=3)

# Check if there are any matching results or if the relevance score is too low
if len(results) == 0 or results[0][1] < 0.7:
    print(f"Unable to find matching results.")

# print(results)


context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
 
# Create prompt template using context and query text
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query_text)

# Initialize OpenAI chat model
model = ChatOpenAI(
  model_name="gpt-3.5-turbo",
  temperature=0,
  top_p=0
)

# Generate response text based on the prompt
response_text = model.predict(prompt)


print(response_text)
