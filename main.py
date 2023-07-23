from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.evaluation import load_evaluator
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

## Step1. Load Resumes
loader = DirectoryLoader("resume_all/", glob="*.pdf", loader_cls=PyPDFLoader)
pages = loader.load_and_split()
# print(pages[0])

## Step2. Load Job Description
url = "https://boards.greenhouse.io/appier/jobs/3446850"
loader = WebBaseLoader(url)
descriptions = loader.aload()
description_text = descriptions[0].page_content

## Step3. Find k resumes fit(similarity) the job description the most
embeddings = OpenAIEmbeddings()
faiss_index = FAISS.from_documents(pages, embeddings)
docs = faiss_index.similarity_search(description_text, k=1)

## Step4. Evaluation and Scores
evaluator = load_evaluator("pairwise_embedding_distance")
print("length:", len(docs))
resumes = []
for doc in docs:
    score = evaluator.evaluate_string_pairs(prediction=doc.page_content, prediction_b=description_text)["score"]
    print(score)
    print(str(doc.metadata["source"]), end="\n\n") #  + ":", doc.page_content[:300]
    resumes.append(doc.metadata['source'])

print(f"Recommand Resume: {resumes}")