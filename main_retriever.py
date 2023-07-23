from langchain.document_loaders import WebBaseLoader
from langchain.retrievers import DocArrayRetriever
from docarray.index import HnswDocumentIndex
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.evaluation import load_evaluator
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
from docarray.typing import NdArray
from docarray import BaseDoc, DocList





## Step1. Load Resumes
loader = DirectoryLoader("resume_all/", glob="*.pdf", loader_cls=PyPDFLoader)
pages = loader.load_and_split()
# print(pages[0])

## Step2. Create MyDoc class as the document schema
class MyDoc(BaseDoc):
    filename: str
    text: str
    text_embedding: NdArray[1536]

## Step3. Add resumes(embedding) to db
embeddings = OpenAIEmbeddings()
docs = DocList[MyDoc](
    [
        MyDoc(
            filename=page.metadata["source"],
            text=page.page_content,
            text_embedding=embeddings.embed_query(page.page_content)
        )
        for page in pages
    ]
)
db = HnswDocumentIndex[MyDoc](work_dir="resume_search")
db.index(docs) # add data to db

# Step4. Create a retriever
retriever = DocArrayRetriever(
    index=db,
    embeddings=embeddings,
    search_field="text_embedding",
    content_field="text",
)

## Step5. Load Job Description
url = "https://boards.greenhouse.io/appier/jobs/3446850"
loader = WebBaseLoader(url)
descriptions = loader.aload()
descriptions_text = descriptions[0].page_content

# Find the relevant documents
docs = retriever.get_relevant_documents(descriptions_text, top_k=1)
resumes = []
for doc in docs:
    resumes.append(doc.metadata['filename'])
print(f"Recommand Resume: {resumes}")


