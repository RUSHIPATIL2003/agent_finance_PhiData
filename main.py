"""Main entry point for the RAG application."""

import sys
from app.config import get_settings
from app.utils.logging import setup_logging
from app.database import init_database, check_connection
from app.ingestion import ingest_document

logger = setup_logging()


def main():
    """Main application entry point."""
    settings = get_settings()
    
    logger.info("Starting Multimodal RAG Chatbot")
    logger.info("Configuration loaded: %s", settings.postgres_host)
    
    # Initialize database
    logger.info("Initializing database...")
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        sys.exit(1)
    
    # Check connection
    if not check_connection():
        logger.error("Database connection check failed")
        sys.exit(1)
    logger.info("Database connection verified")
    
    # Ingest document if needed
    logger.info("Checking document for ingestion...")
    try:
        result = ingest_document(settings.document_path)
        logger.info("Ingestion result: %s", result)
    except Exception as e:
        logger.error("Document ingestion failed: %s", e)
        sys.exit(1)
    
    logger.info("Application ready! Start Streamlit with: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()