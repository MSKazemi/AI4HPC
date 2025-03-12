import os
from parser import parse_document  # File/document parsing

import crawler  # Web crawling
from chunker import TextChunker  # Text chunking
from embedder import TextEmbedder  # Text embedding
from store import ChromaVectorStore  # ChromaDB storage

# Define URLs and file paths
urls = [
    "https://wiki.u-gov.it/confluence/display/SCAIUS/Get+Started#GetStarted-1.Registration",
    "https://wiki.u-gov.it/confluence/display/SCAIUS/Get+Started",
    "https://wiki.u-gov.it/confluence/display/SCAIUS/LEONARDO+User+Guide"
]

# file_paths = [
#     "example.pdf",   # Example PDF file
#     "example.docx",  # Example DOCX file
#     "example.txt"    # Example TXT file
# ]

embedder = TextEmbedder()
store = ChromaVectorStore() # MAYBE YOU NEED TO PASS SOME ARGS...
chunker = TextChunker()

# === Process Web URLs ===
for url in urls:
    print(f"Processing URL: {url}")

    # Step 1: Crawl the web page and extract HTML
    html_content = crawler.fetch_docs([url])[0]

    # Step 2: Save the HTML to a temporary file for parsing
    temp_html_file = "temp.html"
    with open(temp_html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Step 3: Parse extracted HTML to get clean text
    parsed_text = parse_document(temp_html_file)

    # Step 4: Chunk the parsed text
    text_chunks = chunker.chunk_text(parsed_text)

    # Step 5: Generate embeddings for the text chunks
    # embeddings = embedder.embed_texts(text_chunks) # Not necessary for now

    # Step 6: Store in ChromaDB
    store.index_documents(
        texts=text_chunks,  # List of text chunks
        metadata=[{"url": url, "source": url}] * len(text_chunks)  # Corresponding list of metadata dictionaries
    )

    print(f"Stored {url} in ChromaDB.")
    os.remove(temp_html_file)  # Clean up temp file

# # === Process Local Files ===
# for file_path in file_paths:
#     print(f"Processing File: {file_path}")

#     if os.path.exists(file_path):
#         # Step 1: Parse the document
#         parsed_text = parse_document(file_path)

#         # Step 2: Generate embeddings
#         embeddings = embedder.generate_embedding(parsed_text)

#         # Step 3: Store in ChromaDB
#         store.add_document(url=file_path, text=parsed_text, embedding=embeddings)

#         print(f"Stored {file_path} in ChromaDB.")
#     else:
#         print(f"File not found: {file_path}")

print("All documents processed successfully!")

