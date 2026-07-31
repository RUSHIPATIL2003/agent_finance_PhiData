-- PostgreSQL + pgvector initialization script
-- Run this script as a superuser (postgres) to set up the database

-- Create the database if it doesn't exist
SELECT 'CREATE DATABASE rag_agent'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'rag_agent')\gexec

-- Connect to the rag_agent database
\c rag_agent;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create schema for RAG application
CREATE SCHEMA IF NOT EXISTS rag;

-- Set search path
SET search_path TO rag, public;

-- Create documents table
CREATE TABLE IF NOT EXISTS rag.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    file_hash VARCHAR(64) NOT NULL UNIQUE,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(100),
    page_count INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on file_hash for fast duplicate detection
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON rag.documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON rag.documents(filename);

-- Create chunks table with vector embeddings
CREATE TABLE IF NOT EXISTS rag.chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES rag.documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(50) NOT NULL DEFAULT 'text',
    page_number INTEGER,
    section_title VARCHAR(500),
    heading_hierarchy JSONB DEFAULT '[]',
    bbox JSONB,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON rag.chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page_number ON rag.chunks(page_number);
CREATE INDEX IF NOT EXISTS idx_chunks_content_type ON rag.chunks(content_type);

-- Create vector similarity search index (HNSW for better performance)
-- Note: Requires pgvector 0.5.0+
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
    ON rag.chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Alternative IVFFlat index (older pgvector versions)
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat 
--     ON rag.chunks USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

-- Create conversations table for chat history
CREATE TABLE IF NOT EXISTS rag.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    retrieved_chunks UUID[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON rag.conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON rag.conversations(created_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION rag.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
DROP TRIGGER IF EXISTS update_documents_updated_at ON rag.documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON rag.documents
    FOR EACH ROW EXECUTE FUNCTION rag.update_updated_at_column();

DROP TRIGGER IF EXISTS update_chunks_updated_at ON rag.chunks;
CREATE TRIGGER update_chunks_updated_at
    BEFORE UPDATE ON rag.chunks
    FOR EACH ROW EXECUTE FUNCTION rag.update_updated_at_column();

-- Create function for similarity search
CREATE OR REPLACE FUNCTION rag.similarity_search(
    query_embedding VECTOR(384),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 5,
    filter_document_id UUID DEFAULT NULL
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    content_type VARCHAR(50),
    page_number INTEGER,
    section_title VARCHAR(500),
    heading_hierarchy JSONB,
    metadata JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id AS chunk_id,
        c.document_id,
        c.content,
        c.content_type,
        c.page_number,
        c.section_title,
        c.heading_hierarchy,
        c.metadata,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM rag.chunks c
    WHERE (filter_document_id IS NULL OR c.document_id = filter_document_id)
      AND 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Create function for hybrid search (vector + keyword)
CREATE OR REPLACE FUNCTION rag.hybrid_search(
    query_embedding VECTOR(384),
    query_text TEXT,
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 5,
    filter_document_id UUID DEFAULT NULL,
    vector_weight FLOAT DEFAULT 0.7,
    keyword_weight FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    content_type VARCHAR(50),
    page_number INTEGER,
    section_title VARCHAR(500),
    heading_hierarchy JSONB,
    metadata JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id AS chunk_id,
        c.document_id,
        c.content,
        c.content_type,
        c.page_number,
        c.section_title,
        c.heading_hierarchy,
        c.metadata,
        (vector_weight * (1 - (c.embedding <=> query_embedding)) + 
         keyword_weight * ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text))) AS similarity
    FROM rag.chunks c
    WHERE (filter_document_id IS NULL OR c.document_id = filter_document_id)
      AND (1 - (c.embedding <=> query_embedding) > match_threshold OR 
           to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text))
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your user)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA rag TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA rag TO your_app_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA rag TO your_app_user;

-- Verify installation
SELECT 'pgvector version: ' || extversion FROM pg_extension WHERE extname = 'vector';
SELECT 'Database initialization complete!' AS status;