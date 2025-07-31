import os
import time
import logging
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import requests
from requests.auth import HTTPBasicAuth

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def openapi_client():
    """Configure ChatOpenAI client for custom OpenAPI endpoint with basic auth."""
    logger.debug("=== Configuring ChatOpenAI for OpenAPI with Basic Auth ===")

    # Get credentials from environment variables
    username = os.environ.get('OPENAPI_USERNAME', 'CQpWpxxS1ifpoJ5LE5VOQV')
    password = os.environ.get('OPENAPI_PASSWORD', 'uiITiIPq52oQzqF2pzvj8Z')
    api_key = os.environ.get('OPENAPI_KEY', '123')

    logger.debug(f"Using credentials: {username}:*****")
    logger.debug(f"API Key: {api_key}")

    # Create ChatOpenAI client with custom configuration
    chat = ChatOpenAI(
        model="llama-3.1-8b",  # Use the model available at your OpenAPI endpoint
        temperature=0.7,
        max_tokens=32000,
        api_key="123",
        verbose=True,
        base_url="https://llm.ruphus.dev/v1/",  # Replace with your OpenAPI endpoint
        default_headers={
            "Authorization": f"Basic Q1FwV3B4eFMxaWZwb0o1TEU1Vk9RVjp1aUlUaUlQcTUyb1F6cUYycHp2ajha"
        },
    )

    return chat

def openapi_embeddings():
    # Option 1: Add the ls-real-model-name header
    return OpenAIEmbeddings(
        model="llama-3.1-8b",  # This might be ignored by your proxy
        api_key="123",
        base_url="https://llm.ruphus.dev/v1/",
        default_headers={
            "Authorization": f"Basic Q1FwV3B4eFMxaWZwb0o1TEU1Vk9RVjp1aUlUaUlQcTUyb1F6cUYycHp2ajha",
            "ls-real-model-name": "llama3.1-8b"  # or whatever embedding model your server supports
        }
    )


def test_openapi_connection():
    """Test connection to OpenAPI endpoint with basic authentication."""
    logger.debug("=== Testing OpenAPI Connection ===")


    # Test URL (replace with your actual endpoint)
    test_url = "https://llm.ruphus.dev/v1"

    logger.debug(f"Testing connection to: {test_url}")


    try:
        # Make test request with basic auth
        start_time = time.time()
        logger.debug("Sending test request...")

        response = requests.get(
            test_url,
            auth=HTTPBasicAuth(username, password),
            headers={'Accept': 'application/json'},
            verify=False,  # Disable SSL verification for self-signed certs
            timeout=10  # Add timeout for the request
        )

        elapsed_time = time.time() - start_time
        logger.debug(f"Request completed in {elapsed_time:.2f} seconds")

        # Log detailed response information
        logger.debug(f"Status Code: {response.status_code}")
        logger.debug(f"Headers: {dict(response.headers)}")
        logger.debug(f"Encoding: {response.encoding}")
        logger.debug(f"Content type: {response.headers.get('content-type', 'N/A')}")

        if response.status_code == 200:
            logger.debug("✅ Connection successful!")
            try:
                response_json = response.json()
                logger.debug(f"Response JSON: {response_json}")
            except ValueError:
                logger.debug(f"Response body: {response.text}")
        else:
            logger.debug(f"❌ Connection failed with status: {response.status_code}")
            logger.debug(f"Response body: {response.text}")

    except requests.exceptions.Timeout:
        logger.error("❌ Request timed out")
    except requests.exceptions.SSLError as e:
        logger.error(f"❌ SSL Error: {e}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Connection Error: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Request Exception: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected Error: {e}")
    except:
        logger.error("❌ An unknown error occurred")

def chat_with_openapi():
    """Demonstrate chat functionality with OpenAPI endpoint."""
    logger.debug("\n=== Chat with OpenAPI Endpoint ===")

    # Configure client
    chat = openapi_client()

    # Define system message
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant connected to a custom OpenAPI endpoint. "
            "You maintain context from previous messages in the conversation."
        )
    )

    logger.debug("Chat is ready! Type 'exit' to quit")

    # Chat loop
    chat_history = [system_message]
    while True:
        # Get user input
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            logger.debug("Goodbye!")
            break

        # Create human message
        human_message = HumanMessage(content=user_input)
        chat_history.append(human_message)

        # Get AI response
        try:
            logger.debug(f"Sending chat request with input: {user_input}")
            start_time = time.time()

            ai_response = chat(chat_history)
            elapsed_time = time.time() - start_time

            logger.debug(f"Received response in {elapsed_time:.2f} seconds")
            logger.debug(f"Response type: {type(ai_response)}")
            logger.debug(f"Response content: {ai_response.content}")

            ai_message = AIMessage(content=ai_response.content)
            chat_history.append(ai_message)

            logger.debug(f"AI: {ai_response.content}")

        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error("Failed to get response from OpenAPI endpoint")
            logger.error(f"Chat history at time of failure: {chat_history}")
            break

    