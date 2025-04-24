## RAG OpenAI API embeddings FAISS with chunks and stream response

First of all, install all requirements.

> pip install -r requirements.txt

This is a simple RAG example using OpenAI API, with chunks and possible stream responses, using OpenAI for embeddings and vector databases and cosine for knowledge management.

We have:

1. An OpenAI client (API key needed for the user, not uploaded for safety reasons), refactored in a clean OpenAIClient class.
2. RAG. Retrieves information from different documents and answers queries with respect to them.
    - Texts: Extract texts from different PDFs using PdfReader.
    - Chunks: Divide each text in chunks to simulate larger documents for knowledge basis.
    - Embeddings: OpenAI API.
    - Semantic similarity: Cosine between embeddings from knowledge basis and query.