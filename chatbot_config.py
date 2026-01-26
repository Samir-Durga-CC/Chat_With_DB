"""
Chatbot Configuration
Centralized configuration for database agent and main agent
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

class DatabaseConfig:
    """PostgreSQL and MongoDB configuration"""
    
    # PostgreSQL
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', '192.168.10.159')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'alpha-product-samir')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
    
    # MongoDB
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://root:password@10.212.69.113:27020/')
    MONGO_DB = os.getenv('MONGO_DB', 'alpha-kcc')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'conversationhistory')


# =============================================================================
# SCHEMA CONFIGURATION
# =============================================================================

class SchemaConfig:
    """Define which tables/collections to include in schema"""
    
    # PostgreSQL Tables to Include
    # Format: {"database_name": ["table1", "table2", ...]}
    POSTGRES_TABLES = {
        f"{DatabaseConfig.POSTGRES_DB}": [
            "countries",
            "countries_group",
            "product_metas",
            "products",
            "categories"
        ]
    }
    
    # MongoDB Collections to Include
    # Format: {"database_name": ["collection1", "collection2", ...]}
    MONGODB_COLLECTIONS = {
        f"{DatabaseConfig.MONGO_DB}": [
            "countries",
            "excludedproducts",
            "internationallinkages",
            "referencestandards",
            "regulationmetadatas"
        ]
    }
    
    # Columns to EXCLUDE from responses (in addition to UUID columns)
    # These will be filtered out automatically
    EXCLUDED_COLUMNS = [
        # Metadata columns
        "createdAt",
        "updatedAt",
        "createdBy",
        "updatedBy",
        "deletedAt",
        "deletedBy",
        
        # Internal flags
        "showInKnowledgebase",
        "isActive",
        "isDeleted",
        
        # Add more columns here as needed
        # "columnName",
    ]
    
    # Note: UUID columns are automatically detected and excluded
    # No need to list them here


# =============================================================================
# OPENAI CONFIGURATION
# =============================================================================

class OpenAIConfig:
    """OpenAI API configuration"""
    
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    
    # Agent settings
    MAX_ITERATIONS = int(os.getenv('MAX_ITERATIONS', 10))
    STREAMING_DELAY = float(os.getenv('STREAMING_DELAY', 0.01))


# =============================================================================
# VECTOR STORE CONFIGURATION
# =============================================================================

class VectorStoreConfig:
    """Vector store configuration"""
    
    KNOWLEDGEBASE_COLLECTION_PREFIX = "knowledgebase"
    COMPLIANCEWIZARD_COLLECTION_PREFIX = "complianceWizard"
    EXTERNAL_COLLECTION = "External_Knowledgebase"
    
    DEFAULT_SEARCH_LIMIT = 5
    DEFAULT_QUERY_LIMIT = 10


# =============================================================================
# QUERY CONFIGURATION
# =============================================================================

class QueryConfig:
    """Query and response configuration"""
    
    # Default limits
    DEFAULT_QUERY_LIMIT = 10
    MAX_QUERY_LIMIT = 100
    
    # Pagination
    DEFAULT_PAGE_SIZE = 20


# =============================================================================
# EXPORT COMBINED CONFIG
# =============================================================================

class Config:
    """Combined configuration - import this in your modules"""
    
    # Database
    POSTGRES_HOST = DatabaseConfig.POSTGRES_HOST
    POSTGRES_DB = DatabaseConfig.POSTGRES_DB
    POSTGRES_USER = DatabaseConfig.POSTGRES_USER
    POSTGRES_PASSWORD = DatabaseConfig.POSTGRES_PASSWORD
    POSTGRES_PORT = DatabaseConfig.POSTGRES_PORT
    
    MONGO_URI = DatabaseConfig.MONGO_URI
    MONGO_DB = DatabaseConfig.MONGO_DB
    MONGO_DB_NAME = DatabaseConfig.MONGO_DB_NAME
    
    # Schema
    POSTGRES_TABLES = SchemaConfig.POSTGRES_TABLES
    MONGODB_COLLECTIONS = SchemaConfig.MONGODB_COLLECTIONS
    EXCLUDED_COLUMNS = SchemaConfig.EXCLUDED_COLUMNS
    
    # OpenAI
    OPENAI_API_KEY = OpenAIConfig.OPENAI_API_KEY
    OPENAI_MODEL = OpenAIConfig.OPENAI_MODEL
    MAX_ITERATIONS = OpenAIConfig.MAX_ITERATIONS
    
    # Vector Store
    KNOWLEDGEBASE_PREFIX = VectorStoreConfig.KNOWLEDGEBASE_COLLECTION_PREFIX
    COMPLIANCEWIZARD_PREFIX = VectorStoreConfig.COMPLIANCEWIZARD_COLLECTION_PREFIX
    EXTERNAL_COLLECTION = VectorStoreConfig.EXTERNAL_COLLECTION
    
    # Query
    DEFAULT_QUERY_LIMIT = QueryConfig.DEFAULT_QUERY_LIMIT
    MAX_QUERY_LIMIT = QueryConfig.MAX_QUERY_LIMIT
    DEFAULT_PAGE_SIZE = QueryConfig.DEFAULT_PAGE_SIZE