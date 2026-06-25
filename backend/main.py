from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="PDF Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_BASE_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# Loaded once at startup — embedding model is expensive to init
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# session_id -> Chroma instance (in-memory registry)
vector_stores: dict[str, Chroma] = {}


class QuestionRequest(BaseModel):
    session_id: str
    question: str


class SourceDoc(BaseModel):
    page: int | str
    content: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        persist_dir = os.path.join(CHROMA_BASE_DIR, session_id)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        vector_stores[session_id] = vector_store

        return {
            "session_id": session_id,
            "pages": len(docs),
            "chunks": len(chunks),
            "filename": file.filename,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    if req.session_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Session not found. Upload a PDF first.")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    retriever = vector_stores[req.session_id].as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful assistant that answers questions strictly based on the provided document context.
If the answer is not in the context, say "I couldn't find this information in the document."
Be concise and accurate. Do not make up information.

Context:
{context}""",
        ),
        ("human", "{input}"),
    ])

    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    result = chain.invoke({"input": req.question})

    sources = [
        SourceDoc(
            page=doc.metadata.get("page", "N/A"),
            content=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
        )
        for doc in result.get("context", [])
    ]

    return AnswerResponse(answer=result["answer"], sources=sources)
