import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Fast and effective for semantic search

st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>🧠 SHL Assessment Recommendation System</h1>
    <h4 style='text-align: center; color: #ccc;'>Find the best assessments based on your query using AI!</h4>
    <hr style="border: 1px solid #333;">
    """,
    unsafe_allow_html=True
)

# ---- LOAD DATA ----
@st.cache_data
def load_catalog():
    df = pd.read_csv("SHL_catalog.csv")
    df = df.fillna("")
    return df

df = load_catalog()

# ---- BUILD VECTOR DB WITH FAISS ----
@st.cache_resource
def build_vector_db(df):
    # Combine relevant fields for semantic search
    docs = []
    for _, row in df.iterrows():
        content = f"{row['Assessment Name']}. {row['Description']} Skills: {row['Skills']}. Test Type: {row['Test Type']}. Duration: {row['Duration']} minutes."
        docs.append(Document(page_content=content, metadata=row.to_dict()))
    # Embedding model
    embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL_NAME)
    # Prepare texts and metadata for FAISS
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vectordb = FAISS.from_texts(texts, embedding=embedder, metadatas=metadatas)
    return vectordb

vectordb = build_vector_db(df)

query = st.text_area(
    "Job Description or Query",
    placeholder="E.g. I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
)

top_n = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Recommend Assessments"):
    if not query.strip():
        st.warning("Please enter a job description or query.")
    else:
        # Retrieve top-N relevant docs using embeddings
        retriever = vectordb.as_retriever(search_kwargs={"k": top_n})
        retrieved_docs = retriever.get_relevant_documents(query)
        if not retrieved_docs:
            st.info("No relevant assessments found.")
        else:
            # Format output table
            rows = []
            for doc in retrieved_docs:
                meta = doc.metadata
                rows.append({
                    "Assessment Name": f'<a href="{meta["URL"]}" target="_blank">{meta["Assessment Name"]}</a>',
                    "Remote Testing Support": meta["Remote Testing Support"],
                    "Adaptive/IRT Support": meta["Adaptive/IRT"],
                    "Duration": meta["Duration"],
                    "Test Type": meta["Test Type"],
                })
            out_df = pd.DataFrame(rows)
            st.markdown(
                out_df.to_html(escape=False, index=False),
                unsafe_allow_html=True
            )

st.caption("Built with Streamlit · LangChain · HuggingFace Embeddings · FAISS")
