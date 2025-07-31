from tools.embedding_tool import EmbeddingTool
import pdfplumber
import os

def test_document_cleaning():
    """Test the text cleaning functionality of EmbeddingTool on the Venstre program document."""
    
    # Initialize the embedding tool
    embedding_tool = EmbeddingTool()
    
    # Get the path to the PDF document
    current_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(current_dir, "venstre-stortingsprogram-2025.pdf")

    print(f"Reading PDF from: {pdf_path}")
    
    try:
        # Read raw text from PDF
        raw_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            print(f"PDF has {len(pdf.pages)} pages")
            for page_num, page in enumerate(pdf.pages[:5]):  # Read first 5 pages for testing
                page_text = page.extract_text()
                if page_text:
                    raw_text += page_text
                print(f"Processed page {page_num + 1}")
        
        print(f"\nRaw text length: {len(raw_text)} characters")
        print("=" * 60)
        print("FIRST 1000 CHARACTERS OF RAW TEXT:")
        print("=" * 60)
        print(raw_text[:1000])
        
        # Clean the text using the embedding tool
        print("\n" + "=" * 60)
        print("CLEANING TEXT...")
        print("=" * 60)
        
        cleaned_text = embedding_tool.clean_text(raw_text)
        
        print(f"\nCleaned text length: {len(cleaned_text)} characters")
        print("=" * 60)
        print("FIRST 1000 CHARACTERS OF CLEANED TEXT:")
        print("=" * 60)
        print(cleaned_text[:1000])
        
        # Show the difference in length
        print(f"\nText length difference: {len(raw_text) - len(cleaned_text)} characters")
        print(f"Compression ratio: {len(cleaned_text)/len(raw_text):.2%}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_document_cleaning()

