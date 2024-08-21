from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant

# Retrieval Hyperparameters
chunk_size = 1500

# Define embeddings
embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-base-en-v1.5")

print(embeddings)

# Define loader
loader = DirectoryLoader(
    "data/", glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=(chunk_size / 10)
)

texts = text_splitter.split_documents(documents)

print(texts[1])

url = "http://localhost:6333"

qdrant = Qdrant.from_documents(
    texts, embeddings, url=url, prefer_grpc=False, collection_name="cookbook_db"
)

print("Vector DB Successfully Created!")
