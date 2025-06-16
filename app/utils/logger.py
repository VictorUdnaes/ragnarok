import logging

logger = logging.getLogger("RAGService")
logger.setLevel(logging.INFO)  # Set the logging level for your custom logger

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the level for the console handler

# Define a custom log format
formatter = logging.Formatter(" %(message)s")
console_handler.setFormatter(formatter)

# Add the handler to your logger
logger.addHandler(console_handler)

# Suppress logs from external libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)  # Example: Suppress urllib3 logs
logging.getLogger("chroma").setLevel(logging.WARNING)   # Example: Suppress Chroma logs
logging.getLogger().setLevel(logging.WARNING)           # Suppress root logger logs