"""Database connection pool and operations module."""

import contextlib
import logging
from typing import Any, Generator, Optional

from psycopg import Connection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

from app.config import get_settings

logger = logging.getLogger(__name__)


class DatabasePool:
    """Thread-safe PostgreSQL connection pool manager."""

    _instance: Optional["DatabasePool"] = None
    _pool: Optional[ConnectionPool] = None

    def __new__(cls) -> "DatabasePool":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._pool is None:
            self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Initialize the connection pool."""
        settings = get_settings()
        
        conninfo = (
            f"host={settings.postgres_host} "
            f"port={settings.postgres_port} "
            f"dbname={settings.postgres_db} "
            f"user={settings.postgres_user} "
            f"password={settings.postgres_password}"
        )

        self._pool = ConnectionPool(
            conninfo=conninfo,
            min_size=settings.db_pool_min_conn,
            max_size=settings.db_pool_max_conn,
            timeout=settings.db_pool_timeout,
            kwargs={"row_factory": dict_row},
        )
        logger.info(
            "Database connection pool initialized: min=%d, max=%d",
            settings.db_pool_min_conn,
            settings.db_pool_max_conn,
        )

    @property
    def pool(self) -> ConnectionPool:
        """Get the connection pool instance."""
        if self._pool is None:
            self._initialize_pool()
        return self._pool

    @contextlib.contextmanager
    def connection(self) -> Generator[Connection, None, None]:
        """Get a connection from the pool."""
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    @contextlib.contextmanager
    def transaction(self) -> Generator[Connection, None, None]:
        """Get a connection with automatic transaction handling."""
        with self.connection() as conn:
            with conn.transaction():
                yield conn

    def close(self) -> None:
        """Close all connections in the pool."""
        if self._pool:
            self._pool.close()
            self._pool = None
            DatabasePool._instance = None
            logger.info("Database connection pool closed")

    def execute(self, query: str, params: Optional[tuple] = None) -> list[dict[str, Any]]:
        """Execute a query and return results."""
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:
                    return cur.fetchall()
                return []

    def execute_one(self, query: str, params: Optional[tuple] = None) -> Optional[dict[str, Any]]:
        """Execute a query and return a single result."""
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchone()

    def execute_many(self, query: str, params_list: list[tuple]) -> int:
        """Execute a query multiple times with different parameters."""
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, params_list)
                return cur.rowcount


def get_db_pool() -> DatabasePool:
    """Get the database pool singleton."""
    return DatabasePool()


def init_database() -> None:
    """Initialize database schema by running SQL script."""
    pool = get_db_pool()
    settings = get_settings()
    
    sql_path = "sql/init_database.sql"
    try:
        with open(sql_path, "r") as f:
            sql_script = f.read()
        
        # Split by semicolon and execute each statement
        statements = [s.strip() for s in sql_script.split(";") if s.strip()]
        
        with pool.transaction() as conn:
            with conn.cursor() as cur:
                for stmt in statements:
                    if stmt and not stmt.startswith("--"):
                        try:
                            cur.execute(stmt)
                        except Exception as e:
                            logger.warning("Statement failed (may be expected): %s", e)
        
        logger.info("Database schema initialized successfully")
    except FileNotFoundError:
        logger.error("SQL initialization script not found at %s", sql_path)
        raise
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)
        raise


def check_connection() -> bool:
    """Check if database connection is working."""
    try:
        pool = get_db_pool()
        result = pool.execute_one("SELECT 1 as test")
        return result is not None and result.get("test") == 1
    except Exception as e:
        logger.error("Database connection check failed: %s", e)
        return False


def get_document_by_hash(file_hash: str) -> Optional[dict[str, Any]]:
    """Get document by file hash."""
    pool = get_db_pool()
    return pool.execute_one(
        "SELECT * FROM rag.documents WHERE file_hash = %s",
        (file_hash,)
    )


def create_document(
    filename: str,
    file_hash: str,
    file_size: int,
    mime_type: str,
    page_count: int,
    metadata: dict[str, Any]
) -> dict[str, Any]:
    """Create a new document record."""
    pool = get_db_pool()
    return pool.execute_one(
        """
        INSERT INTO rag.documents (filename, file_hash, file_size, mime_type, page_count, metadata)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        RETURNING *
        """,
        (filename, file_hash, file_size, mime_type, page_count, Json(metadata))
    )


def create_chunks_batch(chunks: list[dict[str, Any]]) -> int:
    """Insert multiple chunks in batch."""
    if not chunks:
        return 0
    
    pool = get_db_pool()
    
    query = """
        INSERT INTO rag.chunks (
            document_id, chunk_index, content, content_type, page_number,
            section_title, heading_hierarchy, bbox, metadata, embedding
        ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::vector)
    """

    params_list = [
        (
            c["document_id"],
            c["chunk_index"],
            c["content"],
            c["content_type"],
            c.get("page_number"),
            c.get("section_title"),
            Json(c.get("heading_hierarchy", [])),
            Json(c.get("bbox")) if c.get("bbox") is not None else None,
            Json(c.get("metadata", {})),
            str(c.get("embedding")),
        )
        for c in chunks
    ]
    
    return pool.execute_many(query, params_list)


def similarity_search(
    query_embedding: list[float],
    match_threshold: float = 0.7,
    match_count: int = 5,
    document_id: Optional[str] = None
) -> list[dict[str, Any]]:
    """Perform vector similarity search."""
    pool = get_db_pool()
    
    params = [str(query_embedding), match_threshold, match_count]
    if document_id:
        params.append(document_id)
    else:
        params.append(None)
    
    return pool.execute(
        "SELECT * FROM rag.similarity_search(%s::vector, %s, %s, %s)",
        tuple(params)
    )


def hybrid_search(
    query_embedding: list[float],
    query_text: str,
    match_threshold: float = 0.7,
    match_count: int = 5,
    document_id: Optional[str] = None,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> list[dict[str, Any]]:
    """Perform hybrid vector + keyword search."""
    pool = get_db_pool()
    
    params = [str(query_embedding), query_text, match_threshold, match_count]
    if document_id:
        params.append(document_id)
    else:
        params.append(None)
    params.extend([vector_weight, keyword_weight])
    
    return pool.execute(
        "SELECT * FROM rag.hybrid_search(%s::vector, %s, %s, %s, %s, %s, %s)",
        tuple(params)
    )


def save_conversation(
    session_id: str,
    user_message: str,
    assistant_message: str,
    retrieved_chunks: list[str],
    metadata: dict[str, Any]
) -> dict[str, Any]:
    """Save conversation to database."""
    pool = get_db_pool()
    return pool.execute_one(
        """
        INSERT INTO rag.conversations (session_id, user_message, assistant_message, retrieved_chunks, metadata)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        RETURNING *
        """,
        (session_id, user_message, assistant_message, retrieved_chunks, Json(metadata))
    )


def get_conversation_history(
    session_id: str,
    limit: int = 10
) -> list[dict[str, Any]]:
    """Get conversation history for a session."""
    pool = get_db_pool()
    return pool.execute(
        """
        SELECT * FROM rag.conversations
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (session_id, limit)
    )