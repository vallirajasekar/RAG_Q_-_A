markdown
Copy code
# RAG Document Q&A

This project implements a Retrieval-Augmented Generation (RAG) Document Question and Answering (Q&A) system using Streamlit and Langchain. The application allows users to query documents, providing relevant context-based answers.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Setup](#environment-setup)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Features

- Load and preprocess PDF documents from a directory.
- Create vector embeddings of the documents for efficient retrieval.
- Answer user queries based on the content of the loaded documents.
- Display relevant context from documents to support answers.

## Installation

### Requirements

- Python 3.10
- Streamlit
- Langchain
- FAISS
- dotenv

### Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
Create a new environment:

bash
Copy code
conda create -p venv_gen python==3.10 -y
Activate the environment:

bash
Copy code
conda activate D:\GEN-AI\RAG_DOCUMENT_Q_&_A\venv_gen
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Usage
Ensure your PDF documents are in the research_paper directory.
Run the Streamlit application:
bash
Copy code
streamlit run app.py
Open your web browser and navigate to http://localhost:8501.
Enter your query in the text input and click "Document_Embedding" to prepare the vector database.
After the database is ready, enter your questions to retrieve answers based on the document context.
Environment Setup
Make sure to set up your environment variables for API keys:
GROQ_API_KEY
OPENAI_API_KEY
You can create a .env file in your project directory with the following structure:

makefile
Copy code
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
How It Works
Document Loading: The application uses PyPDFDirectoryLoader to load documents from the specified directory.
Text Splitting: It employs RecursiveCharacterTextSplitter to split documents into manageable chunks for processing.
Vector Embedding: The document chunks are converted into vector embeddings using the OpenAIEmbeddings class, stored in a FAISS vector store for efficient similarity searches.
Question Answering: When a user submits a query, the system retrieves relevant documents based on the vector embeddings and uses the ChatGroq model to generate answers.
Contributing
Contributions are welcome! Please fork the repository and create a pull request for any improvements or bug fixes.