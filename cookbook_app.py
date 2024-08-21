from langchain.prompts import PromptTemplate
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import json

cookbook_app = FastAPI()

templates = Jinja2Templates(directory="templates")
cookbook_app.mount("/static", StaticFiles(directory="static"), name="static")

local_llm = "mistral-7b-instruct-v0.2.Q5_K_M.gguf"

# Configure LLM
# Setting n_ctx solved the problem of premature termination
llm = LlamaCpp(
    model_path=local_llm, temperature=0.3, max_tokens=2048, top_p=0.9, n_ctx=4096
)

print("LLM Initialized....")

prompt_template = """You are a helpful cooking assistant, you will be prompted with a simple question about
a recipe. Answer the question which is given below using the context provided to you.

Context: {context}
Question: {question}
"""

embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-base-en-v1.5")

url = "http://localhost:6333"

client = QdrantClient(url=url, prefer_grpc=False)

db = Qdrant(client=client, embeddings=embeddings, collection_name="cookbook_db")

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Use top k chunks as context
retriever = db.as_retriever(search_kwargs={"k": 1})


# Fast API
@cookbook_app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@cookbook_app.post("/get_response")
async def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )
    response = qa(query)
    answer = response["result"]
    for doc in response["source_documents"]:
        print(doc.page_content)
    # source_document = response["source_documents"][0].page_content
    # doc = response["source_documents"][0].metadata["source"]
    response_data = jsonable_encoder(json.dumps({"answer": answer}))

    res = Response(response_data)
    return res
