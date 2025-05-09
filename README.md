## RAG OpenAI API with different documents separated in chunks, with a manual knowledge basis in JSON files, and semantic similarity using cosine

First of all, install all requirements.

> pip install -r requirements.txt

This is a simple RAG example using OpenAI API. Details:

1. An OpenAI client (API key needed for the user, not uploaded for safety reasons), refactored in a clean OpenAIClient class.
2. RAG. Retrieves information from different documents and answers queries with respect to them.
    - Texts: Extract texts from different PDFs using PdfReader.
    - Chunks: Divide each text in chunks to simulate larger documents for knowledge basis.
    - Embeddings: OpenAI API.
    - Semantic similarity: Cosine between embeddings from knowledge basis and query.