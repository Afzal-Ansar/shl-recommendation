import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# Initialize Embedding Model
embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Initialize Output Parser
parser = StrOutputParser()

# Define Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Use the provided context to answer the query.\n"
    "If unsure, state that you don't know. Be concise and factual.\n"
    "Context: {document_context}\n"
    "Query: {user_query}\n"
    "Answer:"
)

# Initialize LLM
llm = ChatGroq(api_key="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",model="llama-3.3-70b-versatile")

# Streamlit App Title
st.title("🎥 Chat with a YouTube Video")
st.image(r"C:\Users\99ash\Downloads\unnamed.jpg",width=300)

# User input for YouTube URL
yt_url = st.text_input("Enter YouTube video URL:")

# Function to Load and Process Video
def load_video_data(url):
    try:
        # Try loading English transcript first
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,
            language=["en"],       # Try English first
            translation="en"       # Allow translation if needed
        )
        docs = loader.load()
    except Exception as e:
        st.warning("English subtitles not available. Trying Hindi subtitles with translation...")
        # Try Hindi if English is not available
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,
            language=["hi"],
            translation="en"
        )
        docs = loader.load()

    # Split Documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # Create FAISS Vector Store
    vectorstore = FAISS.from_documents(splits, embed)
    retriever = vectorstore.as_retriever()

    # Build the full Chain
    chain = (
        RunnableParallel({"document_context": retriever, "user_query": RunnablePassthrough()})
        | prompt
        | llm
        | parser
    )
    return chain

# Main Streamlit App Logic
if yt_url:
    try:
        chain = load_video_data(yt_url)

        prompt_text = st.text_input("Ask something about the video...")

        if prompt_text:
            answer = chain.invoke(prompt_text)
            st.markdown("**Answer:**")
            st.success(answer)

    except Exception as e:
        st.error(f"Error: {str(e)}")
