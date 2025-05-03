import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import uvicorn
import os

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# ---- CONFIG ----
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CATALOG_CSV = "SHL_catalog.csv"

# ---- LOAD DATA ----
def load_catalog():
    df = pd.read_csv(r"C:\Users\99ash\Downloads\SHL_catalog.csv")
    df = df.fillna("")
    return df

df = load_catalog()

# ---- BUILD VECTOR DB ----
def build_vector_db(df):
    docs = []
    for _, row in df.iterrows():
        content = f"{row['Assessment Name']}. {row['Description']} Skills: {row['Skills']}. Test Type: {row['Test Type']}. Duration: {row['Duration']} minutes."
        docs.append(Document(page_content=content, metadata=row.to_dict()))
    embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL_NAME)
    vectordb = Chroma.from_documents(docs, embedding=embedder)
    return vectordb

vectordb = build_vector_db(df)

# ---- FASTAPI APP ----
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API for recommending SHL assessments based on job description or query.",
    version="1.0.0"
)

class Assessment(BaseModel):
    assessment_name: str
    url: str
    remote_testing_support: str
    adaptive_irt_support: str
    duration: int
    test_type: str

class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[Assessment]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(query: str = Query(..., description="Job description or natural language query"),
              top_n: int = Query(10, ge=1, le=10, description="Number of recommendations to return (max 10)")):
    retriever = vectordb.as_retriever(search_kwargs={"k": top_n})
    retrieved_docs = retriever.get_relevant_documents(query)
    recs = []
    for doc in retrieved_docs:
        meta = doc.metadata
        recs.append(Assessment(
            assessment_name=meta["Assessment Name"],
            url=meta["URL"],
            remote_testing_support=meta["Remote Testing Support"],
            adaptive_irt_support=meta["Adaptive/IRT"],
            duration=int(meta["Duration"]),
            test_type=meta["Test Type"]
        ))
    return RecommendationResponse(query=query, recommendations=recs)

# ---- MAIN ----
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
