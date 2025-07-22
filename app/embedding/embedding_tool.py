import re
import PyPDF2
from langchain.docstore.document import Document
import spacy

class EmbeddingTool():
    def __init__(self):
        self.nlp = spacy.load("nb_core_news_sm")  # Load once to avoid repeated loading

    def embed_document_with_chunking(self, text: str, chunk_size: int = 1000) -> list[str]:
        """Splits the text into chunks of specified size."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    def embed_document_with_pattern(self, text: str, pattern: str) -> list[str]:
        regex = re.compile(pattern)
        return regex.findall(text)
    
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