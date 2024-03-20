from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain.embeddings import CohereEmbeddings
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import Cohere
import os

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Get COHERE_API_KEY from environment variable
cohere_api_key = os.getenv("COHERE_API_KEY")

# Check if COHERE_API_KEY is available
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY environment variable is not set.")

# Initialize Cohere Embeddings
try:
    embeddings = CohereEmbeddings(model="large", cohere_api_key=cohere_api_key)
    print("Cohere Embeddings initialized successfully.")
except ValueError as e:
    print(f"Error initializing Cohere Embeddings: {e}")


# Create temporary folder if it doesn't exist
if not os.path.exists("./tempfolder"):
    os.makedirs("./tempfolder")

# File paths
pdf_file_path = 'tempfolder/pmt.pdf' 
pdf_file_path2 = 'tempfolder/Chemicals.pdf' 
pdf_file_path3 = 'tempfolder/agrigpt.pdf' 

# Initialize Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=10)

# PDF Loader function
def PDF_loader(document):
    loader = OnlinePDFLoader(document)
    documents = loader.load()
    prompt_template = """ 
    You are an AI Chatbot developed to help users by suggesting eco-friendly farming methods, alternatives to chemical pesticides and fertilizers, and maximizing profits. Use the following pieces of context to answer the question at the end. Greet Users!!
    {context}

    {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    texts = text_splitter.split_documents(documents)
    global db
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    global qa
    qa = RetrievalQA.from_chain_type(
        llm=Cohere(
            model="command-xlarge-nightly",
            temperature=0.98,
            cohere_api_key=cohere_api_key,
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs=chain_type_kwargs,
    )
    return "Ready"

# Load PDF
PDF_loader(pdf_file_path)

# Initialize session state
app.config['SECRET_KEY'] = 'your_secret_key'

# Route to render index.html template
@app.route('/')
def index():
    return render_template('index.html', name="AgroIntel- Chatbot")

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    # Handle form submission here
    return 'Form submitted successfully'

if __name__ == '__main__':
    app.run(debug=True)
