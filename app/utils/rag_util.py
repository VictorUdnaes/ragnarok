from langchain.load import dumps, loads
import warnings

@staticmethod
def get_unique_union(documents: list[list]):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
            # Get unique documents
            unique_docs = list(set(flattened_docs))
            # Return
            return [loads(doc) for doc in unique_docs]
    except Exception as e:
        print(f"Error in get_unique_union: {e}")
        return []
    
def sanitize_response(response):
    """Clean up the LLM response, handling DeepSeek R1's thinking tags"""
    try:
        # Ensure the response is a string
        if not isinstance(response, str):
            return str(response)

        # Find the last part after </think> for DeepSeek R1 models
        if "</think>" in response:
            answer = response.split("</think>")[-1].strip()
            return answer if answer else "No answer found after thinking section."

        # If </think> is not found, return the full response
        return response.strip()
        
    except Exception as e:
        return f"Error processing response: {e}"