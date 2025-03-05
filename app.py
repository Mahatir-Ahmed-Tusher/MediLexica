import os
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Function to initialize the LLM
def initial_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",  # Hardcoded API key
        model_name="llama-3.3-70b-versatile"
    )
    return llm

# Function to create the ChromaDB
def create_db():
    pdf_path = "Medilexica.pdf"  # Ensure the PDF is in the root directory

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"🚨 PDF file not found at: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db_path = "chroma_db"
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
    # vector_db.persist()  # No longer needed in Chroma 0.4.x

    print("ChromaDB Created and Data Saved!")
    return vector_db

# Function to set up the QA chain
def setup_qachain(vector_db, llm):
    retriever = vector_db.as_retriever()

    prompt_template = """You are Medilexica, a medical dictionary and assistant. Your role is to provide accurate, detailed, and empathetic responses to medical queries. Always cite the source of your information, specifically the book "Medilexica". Include examples where applicable. If something is not available in Medilexica, you answer it from your own database.

    Context: {context}
    User: {question}
    Medilexica: """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True  # Enable source citation
    )

    return qa_chain

# Initialize the LLM and ChromaDB
llm = initial_llm()
db_path = "chroma_db"

if not os.path.exists(db_path):
    print("🔄 Creating ChromaDB...")
    vector_db = create_db()
else:
    print("🔄 Loading Existing ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

qa_chain = setup_qachain(vector_db, llm)

# Gradio Interface
def medilexica_query(query):
    try:
        # Get response and source documents
        result = qa_chain({"query": query})
        response = result["result"]
        source_docs = result["source_documents"]

        # Format response with sources
        formatted_response = f"Medilexica: {response}\n\nSources:\n"
        for i, doc in enumerate(source_docs, 1):
            formatted_response += f"{i}. Page {doc.metadata['page'] + 1} of 'Medilexica.pdf': {doc.page_content[:200]}...\n"

        return formatted_response
    except Exception as e:
        return f"Error: {e}"

# Gradio App
with gr.Blocks(title="MediLexica - A Part of EarlyMed Project") as app:
    # Add logo at the top center
    with gr.Row():
        gr.Image("logo.png", scale=1, show_label=False, container=False, width=500)  # Set width directly

    gr.Markdown("# 🩺 **MediLexica**")
    gr.Markdown("### Your Medical Dictionary Assistant")
    gr.Markdown("We know you might feel dumb when your doctor pronounces alien medical terms in front of you, and you feel the urge to be a lil bit smart. Well, ask any medical-related vocabulary, and MediLexica will provide accurate, detailed answers with citations from the **Medilexica** book. MediLexica consists of 3 resourceful books: Webster's New World Medical Dictionary, Dictionary of Medical Terms, and AMA Glossary of Medical Terms.")

    with gr.Row():
        query_input = gr.Textbox(label="Enter your medical-related vocabulary", placeholder="e.g., What is the treatment for severe aortic stenosis?")
        output = gr.Textbox(label="MediLexica's Response", lines=10, interactive=False)

    submit_button = gr.Button("Ask MediLexica")
    submit_button.click(fn=medilexica_query, inputs=query_input, outputs=output)

    gr.Markdown("---")
    gr.Markdown("**Note:** MediLexica is part of the **EarlyMed Project**. It's a medical dictionary for patients who sometimes want to get the hang of the stuff their doc says. BTW, always consult a healthcare professional for medical advice.")

# Launch the Gradio App
app.launch()