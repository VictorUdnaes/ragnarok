import re
import PyPDF2
from langchain.docstore.document import Document
from embedding_tool import EmbeddingTool

book_path ="test.pdf"
embedder = EmbeddingTool()

text = embedder.get_document_as_text(book_path)
quotes = embedder.embed_document_with_pattern(
    text=text, 
    pattern=r'Venstre (?:vil|ønsker)[^.]*\.'
)

print(f"Found {len(quotes)} quotes matching the pattern:")
for i, quote in enumerate(quotes, 1):
    print(f"{i}: {quote.strip()}")