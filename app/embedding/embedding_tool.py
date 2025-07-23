import re
import PyPDF2
from langchain.docstore.document import Document
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter

class EmbeddingTool():
    def __init__(self):
        self.nlp = spacy.load("nb_core_news_sm")  # Load once to avoid repeated loading

    def create_chunks_from_document(self, document_path: str, chunk_size: int = 1000) -> list[Document]:
        text_from_document = self.get_document_as_text(document_path)
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=200
            )
        chunks = text_splitter.split_text(text=text_from_document)
        document_chunks = [Document(page_content=chunk) for chunk in chunks]
        
        return document_chunks
    
    def create_chunks_from_pattern(self, text: str, pattern: str) -> list[Document]:
        regex = re.compile(pattern)
        chunks = regex.findall(text)
        document_chunks = [Document(page_content=chunk) for chunk in chunks]

        return document_chunks

    
    def pre_clean_for_spacy(self, text: str) -> str:
        patterns = [
            # Replace tabs with spaces
            (r'\t', ' '),
            # Replace multiple newlines with single newline
            (r'\n\s*\n', '\n'),
            # Join words split by newlines
            (r'(\w)\n(\w)', r'\1\2'),
            # Replace newlines with spaces
            (r'\n', ' '),
            # Remove spaces before periods
            (r'\s+\.', '.'),
            # Remove spaced dashes
            (r'\s-\s', ''),
            # Number followed by capital letter (like "2020Venstres" -> "2020 Venstres")
            (r'(\d)([A-ZÆØÅ][a-zæøå])', r'\1 \2'),
            # Lowercase letter followed by capital (but not common abbreviations)
            (r'([a-zæøå])([A-ZÆØÅ][a-zæøå]{2,})', r'\1 \2'),
            # Colon followed immediately by capital letter
            (r':([A-ZÆØÅ])', r': \1'),
            # Period followed by number and capital (like ".2AnsvarAlle")  
            (r'\.(\d+)([A-ZÆØÅ][a-zæøå])', r'. \1. \2'),
            # Clean up multiple spaces
            (r' +', ' '),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def add_periods_with_spacy(self, text) -> str:
        """Use spaCy to add periods at sentence boundaries."""
        doc = self.nlp(text)
        result = ""
        
        for sent in doc.sents:
            sentence = sent.text.strip()
            if sentence and not sentence.endswith(('.', '!', '?', ':')):
                sentence += '.'
            result += sentence + ' '
        
        return result.strip()

    def clean_text(self, text: str) -> str:
        text = self.pre_clean_for_spacy(text)
        text = self.add_periods_with_spacy(text)
        text = re.sub(r' +', ' ', text)  # Remove any remaining multiple spaces
        
        return text
    
    def get_document_as_text(self, document_path) -> str:
        with open(document_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            full_text = " ".join([page.extract_text() for page in pdf_reader.pages])

        return full_text