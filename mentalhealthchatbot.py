import os
import gradio as gr
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq  # Corrected import

def initialize_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key="YOUR API KEY HERE",
        model_name="llama-3.3-70b-versatile"  # Corrected model name
    )

def create_vector_db():
    loader = DirectoryLoader("D:/GenAI/My Project/", glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')

    vector_db.persist()
    print('ChromaDB created and data saved')
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """ You are a compassionate Mental Health Chatbot. Respond thoughtfully to the following question:
    {context}
    User:{question}
    Chatbot:"""

    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Load LLM and Database
print("Initializing Chatbot...")
llm = initialize_llm()
db_path = "D:/GenAI/My Project/chroma_db"

if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

qa_chain = setup_qa_chain(vector_db, llm)

# Gradio UI function
def chatbot_response(user_query):
    if user_query.lower() == "exit":
        return "Take Care of Yourself, Goodbye!"
    response = qa_chain.run(user_query)
    return response

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Compassionate Mental Health Chatbot")
    with gr.Row():
        input_text = gr.Textbox(label="You:", placeholder="Ask me anything...")
        output_text = gr.Textbox(label="Chatbot:", interactive=False)
    
    submit_btn = gr.Button("Send")
    
    submit_btn.click(chatbot_response, inputs=[input_text], outputs=[output_text])

# Launch Gradio UI
demo.launch(share=True)
