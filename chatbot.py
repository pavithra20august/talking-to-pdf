
# ✅ LangChain 
# ✅ FAISS 
# ✅ HuggingFace Transformers 
# ✅ SentenceTransformers 
# ✅ PyPDF 
# ✅ All on CPU

from typing import ClassVar, Optional
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import Runnable
from transformers import pipeline

# Sentence-transformer embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([Document(page_content=text)])
    return FAISS.from_documents(docs, embedding_model)

# ✅ Fixed: CustomLLM for use with LangChain & Pydantic v2
class CustomLLM(LLM, Runnable):
    model_name: ClassVar[str] = "google/flan-t5-base"
    pipe: Optional[object] = None  # Needed for Pydantic compatibility

    def __init__(self):
        super().__init__()
        object.__setattr__(self, 'pipe', pipeline("text2text-generation", model=self.model_name))

    def _call(self, prompt: str, stop=None):
        return self.pipe(prompt, max_new_tokens=256)[0]["generated_text"]

    @property
    def _llm_type(self) -> str:
        return "custom"

def get_qa_chain(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=CustomLLM(),
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        return_source_documents=False
    )