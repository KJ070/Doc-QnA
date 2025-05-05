import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer
import torch
import os

# Check if sentencepiece is installed
try:
    import sentencepiece
except ImportError:
    st.error("The `sentencepiece` library is required but not installed. Please install it using `pip install sentencepiece`.")
    st.stop()

# Load LLaMA model and tokenizer with LangChain compatibility
@st.cache_resource
def load_llama_model():
    model_name = "huggyllama/llama-7b"  # Replace with the correct model path
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Create a Hugging Face text generation pipeline
    text_generation_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1
    )

    # Wrap in HuggingFacePipeline to make it LangChain-compatible
    langchain_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    return langchain_llm

# Define a prompt template for LLaMA
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

# Load and preprocess different types of documents
def load_and_preprocess_document(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    temp_file_path = f"temp_file.{file_extension}"

    # Save file temporarily
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load document based on file type
    if file_extension == "txt":
        loader = TextLoader(temp_file_path)
    elif file_extension == "pdf":
        loader = PyMuPDFLoader(temp_file_path)
    elif file_extension == "docx":
        loader = UnstructuredWordDocumentLoader(temp_file_path)
    else:
        st.error("Unsupported file format. Please upload a TXT, PDF, or DOCX file.")
        return None

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Remove temp file
    os.remove(temp_file_path)

    return texts

# Create embeddings and vector store
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# Initialize the RAG-based QA chain
def initialize_qa_chain(vector_store, llm):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

# Function to answer questions using the QA chain
def answer_question(qa_chain, question):
    result = qa_chain({"query": question})
    return result["result"]

# Function to summarize the document
def summarize_document(qa_chain):
    summary_prompt = "Summarize the document in a few sentences."
    return answer_question(qa_chain, summary_prompt)

# Define tools for the agent
def setup_tools(qa_chain):
    tools = [
        Tool(
            name="Document QA",
            func=lambda query: answer_question(qa_chain, query),
            description="Useful for answering questions based on the provided document."
        ),
        Tool(
            name="Document Summarization",
            func=lambda _: summarize_document(qa_chain),
            description="Useful for summarizing the provided document."
        )
    ]
    return tools

# Initialize the agent
def initialize_custom_agent(tools, llm):
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Use a zero-shot reasoning agent
        verbose=True  # Set to True to see the agent's thought process
    )
    return agent

# Streamlit app
def main():
    st.title("QnA")
    st.write("Upload a document and ask questions or request tasks!")

    # Load LLaMA model
    llm = load_llama_model()

    # Upload document
    uploaded_file = st.file_uploader("Upload a document (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])
    
    if uploaded_file is not None:
        texts = load_and_preprocess_document(uploaded_file)

        if texts:
            # Create vector store
            vector_store = create_vector_store(texts)

            # Initialize QA chain
            qa_chain = initialize_qa_chain(vector_store, llm)

            # Set up tools for the agent
            tools = setup_tools(qa_chain)

            # Initialize the agent
            agent = initialize_custom_agent(tools, llm)

            # User input
            user_input = st.text_input("Ask a question or request a task (e.g., 'summarize the document'):")
            if user_input:
                if user_input.lower() == "summarize":
                    summary = summarize_document(qa_chain)
                    st.write("**Summary:**")
                    st.write(summary)
                else:
                    response = agent.run(user_input)
                    st.write("**Response:**")
                    st.write(response)

if __name__ == "__main__":
    main()
