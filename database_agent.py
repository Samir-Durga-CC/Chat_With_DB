"""
Database Agent - Handles all database operations with dynamic schema
Optimized for low latency with streaming support
"""

import os
import json
import logging
from typing import Dict, Any, List, Set
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient
from bson import json_util
from openai import AsyncOpenAI

from chatbot_config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA DISCOVERY FUNCTIONS (For Initial Setup)
# =============================================================================

def discover_postgres_schema():
    """
    Discover all tables and their columns in PostgreSQL
    Run this once to see available tables, then update chatbot_config.py
    """
    try:
        conn = psycopg2.connect(
            host=Config.POSTGRES_HOST,
            database=Config.POSTGRES_DB,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD,
            port=Config.POSTGRES_PORT
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get all tables
        cursor.execute("""
            SELECT 
                schemaname,
                tablename
            FROM pg_tables
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY schemaname, tablename;
        """)
        
        tables = cursor.fetchall()
        
        print("\n" + "="*80)
        print("POSTGRESQL SCHEMA DISCOVERY")
        print("="*80)
        print(f"\nDatabase: {Config.POSTGRES_DB}")
        print(f"Tables found: {len(tables)}\n")
        
        table_list = []
        for table in tables:
            schema = table['schemaname']
            table_name = table['tablename']
            full_name = f'"{schema}"."{table_name}"'
            table_list.append(table_name)
            
            # Get columns for this table
            cursor.execute(f"""
                SELECT 
                    column_name,
                    udt_name as data_type,
                    is_nullable
                FROM information_schema.columns
                WHERE table_schema = '{schema}'
                AND table_name = '{table_name}'
                ORDER BY ordinal_position;
            """)
            
            columns = cursor.fetchall()
            
            print(f"\nTable: {full_name}")
            print(f"Columns: {len(columns)}")
            for col in columns[:10]:  # Show first 10 columns
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                print(f"  - {col['column_name']} ({col['data_type']}) {nullable}")
            
            if len(columns) > 10:
                print(f"  ... and {len(columns) - 10} more columns")
        
        cursor.close()
        conn.close()
        
        # Print formatted list for config file
        print("\n" + "="*80)
        print("COPY THIS TO chatbot_config.py -> SchemaConfig.POSTGRES_TABLES:")
        print("="*80)
        print(f'"{Config.POSTGRES_DB}": {json.dumps(table_list, indent=4)}')
        print("\n")
        
    except Exception as e:
        logger.error(f"PostgreSQL discovery failed: {e}")


def discover_mongodb_schema():
    """
    Discover all collections in MongoDB
    Run this once to see available collections, then update chatbot_config.py
    """
    try:
        client = MongoClient(Config.MONGO_URI)
        db = client[Config.MONGO_DB]
        
        collections = db.list_collection_names()
        
        print("\n" + "="*80)
        print("MONGODB SCHEMA DISCOVERY")
        print("="*80)
        print(f"\nDatabase: {Config.MONGO_DB}")
        print(f"Collections found: {len(collections)}\n")
        
        for collection_name in collections:
            collection = db[collection_name]
            count = collection.count_documents({})
            
            print(f"\nCollection: {collection_name}")
            print(f"Document count: {count}")
            
            # Get sample document to show fields
            sample = collection.find_one()
            if sample:
                print("Sample fields:")
                for key in list(sample.keys())[:15]:  # Show first 15 fields
                    value_type = type(sample[key]).__name__
                    print(f"  - {key} ({value_type})")
                
                if len(sample.keys()) > 15:
                    print(f"  ... and {len(sample.keys()) - 15} more fields")
        
        client.close()
        
        # Print formatted list for config file
        print("\n" + "="*80)
        print("COPY THIS TO chatbot_config.py -> SchemaConfig.MONGODB_COLLECTIONS:")
        print("="*80)
        print(f'"{Config.MONGO_DB}": {json.dumps(collections, indent=4)}')
        print("\n")
        
    except Exception as e:
        logger.error(f"MongoDB discovery failed: {e}")


# Uncomment these lines to run schema discovery
# if __name__ == "__main__":
#     discover_postgres_schema()
#     discover_mongodb_schema()


# =============================================================================
# DATABASE SERVICES WITH DYNAMIC SCHEMA
# =============================================================================

class PostgresService:
    """PostgreSQL operations with dynamic schema and UUID filtering"""
    
    def __init__(self):
        self.config = {
            'host': Config.POSTGRES_HOST,
            'database': Config.POSTGRES_DB,
            'user': Config.POSTGRES_USER,
            'password': Config.POSTGRES_PASSWORD,
            'port': Config.POSTGRES_PORT
        }
        self.conn = None
        self.cursor = None
        self._schema_cache = None
        self._uuid_columns_cache: Set[str] = set()
        self._connect()
        self._discover_uuid_columns()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("PostgreSQL connected")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise
    
    def _discover_uuid_columns(self):
        """Automatically discover all UUID type columns"""
        try:
            # Get all columns with UUID data type
            self.cursor.execute("""
                SELECT DISTINCT column_name
                FROM information_schema.columns
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                AND udt_name = 'uuid'
            """)
            
            uuid_cols = self.cursor.fetchall()
            self._uuid_columns_cache = {row['column_name'] for row in uuid_cols}
            
            logger.info(f"Discovered {len(self._uuid_columns_cache)} UUID columns")
            
        except Exception as e:
            logger.error(f"UUID column discovery failed: {e}")
            self._uuid_columns_cache = set()
    
    def _get_excluded_columns(self) -> Set[str]:
        """Get combined set of excluded columns (UUID + configured)"""
        excluded = self._uuid_columns_cache.copy()
        excluded.update(Config.EXCLUDED_COLUMNS)
        return excluded
    
    def _filter_columns(self, data: List[Dict]) -> List[Dict]:
        """Remove excluded columns from results"""
        if not data:
            return data
        
        excluded_columns = self._get_excluded_columns()
        
        filtered_data = []
        for row in data:
            filtered_row = {
                k: v for k, v in row.items() 
                if k not in excluded_columns
            }
            filtered_data.append(filtered_row)
        
        return filtered_data
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute SQL query with column filtering"""
        try:
            if self.conn.closed:
                self._connect()
            
            self.cursor.execute(query)
            
            if self.cursor.description:
                rows = self.cursor.fetchall()
                data = [dict(row) for row in rows]
                
                # Filter excluded columns
                filtered_data = self._filter_columns(data)
                
                logger.info(f"Query executed: {len(filtered_data)} rows")
                return {
                    'success': True,
                    'data': filtered_data,
                    'count': len(filtered_data)
                }
            else:
                self.conn.commit()
                return {
                    'success': True,
                    'message': f'{self.cursor.rowcount} rows affected'
                }
                
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def get_schema(self):
        """Export database schema"""
        schema_query = """
        SELECT 
            'CREATE TABLE "' || schemaname || '"."' || tablename || '" (' ||
            array_to_string(
                array_agg(
                    '"' || column_name || '" ' || udt_name ||
                    CASE  
                        WHEN character_maximum_length IS NOT NULL 
                        THEN '(' || character_maximum_length || ')'
                        ELSE ''
                    END ||
                    CASE  
                        WHEN is_nullable = 'NO' THEN ' NOT NULL'
                        ELSE ''
                    END
                    ORDER BY ordinal_position
                ),
                ', '
            ) || ');' as schema_def
        FROM (
            SELECT 
                c.column_name,
                c.is_nullable,
                c.character_maximum_length,
                t.schemaname,
                t.tablename,
                c.udt_name,
                c.ordinal_position
            FROM pg_tables t
            JOIN information_schema.columns c 
                ON t.schemaname = c.table_schema 
                AND t.tablename = c.table_name
            WHERE t.schemaname NOT IN ('pg_catalog', 'information_schema')
        ) t
        GROUP BY schemaname, tablename
        ORDER BY schemaname, tablename;
        """
        
        try:
            self.cursor.execute(schema_query)
            results = self.cursor.fetchall()
            
            schema_output = '-- PostgreSQL Database Schema\n\n'
            for row in results:
                schema_output += row['schema_def'] + '\n\n'
            
            return schema_output
        except Exception as e:
            raise Exception(f"Schema export failed: {e}")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


class MongoService:
    """MongoDB operations with dynamic schema and field filtering"""
    
    def __init__(self):
        self.client = MongoClient(Config.MONGO_URI)
        self.db = self.client[Config.MONGO_DB]
        self._schema_cache = None
        self._uuid_fields_cache: Set[str] = set()
        logger.info("MongoDB connected")
        self._discover_uuid_fields()
    
    def _discover_uuid_fields(self):
        """Discover fields that look like UUIDs or ObjectIds"""
        try:
            db_name = Config.MONGO_DB
            collection_names = Config.MONGODB_COLLECTIONS.get(db_name, [])
            
            uuid_fields = set()
            
            for collection_name in collection_names:
                if collection_name not in self.db.list_collection_names():
                    continue
                
                collection = self.db[collection_name]
                sample = collection.find_one()
                
                if sample:
                    for key, value in sample.items():
                        # Check if field is ObjectId or looks like UUID
                        if key == '_id' or 'Id' in key or 'id' == key.lower():
                            uuid_fields.add(key)
            
            self._uuid_fields_cache = uuid_fields
            logger.info(f"Discovered {len(uuid_fields)} UUID-like fields in MongoDB")
            
        except Exception as e:
            logger.error(f"MongoDB UUID field discovery failed: {e}")
            self._uuid_fields_cache = set()
    
    def _get_excluded_fields(self) -> Set[str]:
        """Get combined set of excluded fields"""
        excluded = self._uuid_fields_cache.copy()
        excluded.update(Config.EXCLUDED_COLUMNS)
        excluded.add('_id')  # Always exclude MongoDB _id
        return excluded
    
    def _filter_fields(self, data: Any) -> Any:
        """Remove excluded fields from MongoDB results"""
        if isinstance(data, list):
            return [self._filter_fields(item) for item in data]
        
        if isinstance(data, dict):
            excluded_fields = self._get_excluded_fields()
            return {
                k: self._filter_fields(v) 
                for k, v in data.items() 
                if k not in excluded_fields
            }
        
        return data
    
    def execute(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MongoDB operation with field filtering"""
        try:
            collection_name = operation.get('collection')
            if not collection_name:
                return {'success': False, 'error': 'Collection name required'}
            
            collection = self.db[collection_name]
            action = operation.get('action')
            params = operation.get('params', {})
            
            if action == 'find':
                query = params.get('query', {})
                limit = params.get('limit', Config.DEFAULT_QUERY_LIMIT)
                projection = params.get('projection')
                
                cursor = collection.find(query, projection).limit(limit)
                results = list(cursor)
                
                # Filter excluded fields
                filtered_results = self._filter_fields(results)
                
                logger.info(f"MongoDB find: {len(filtered_results)} documents")
                return {
                    'success': True,
                    'data': json.loads(json_util.dumps(filtered_results)),
                    'count': len(filtered_results)
                }
            
            elif action == 'countDocuments':
                query = params.get('query', {})
                count = collection.count_documents(query)
                
                return {
                    'success': True,
                    'count': count
                }
            
            elif action == 'aggregate':
                pipeline = params.get('pipeline', [])
                results = list(collection.aggregate(pipeline))
                
                # Filter excluded fields
                filtered_results = self._filter_fields(results)
                
                return {
                    'success': True,
                    'data': json.loads(json_util.dumps(filtered_results)),
                    'count': len(filtered_results)
                }
            
            elif action == 'findOne':
                query = params.get('query', {})
                projection = params.get('projection')
                result = collection.find_one(query, projection)
                
                # Filter excluded fields
                filtered_result = self._filter_fields(result)
                
                return {
                    'success': True,
                    'data': json.loads(json_util.dumps(filtered_result)) if filtered_result else None
                }
            
            else:
                return {
                    'success': False,
                    'error': f'Unsupported action: {action}'
                }
                
        except Exception as e:
            logger.error(f"MongoDB operation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def get_schema(self) -> str:
        """Dynamically generate schema for configured collections only"""
        if self._schema_cache:
            return self._schema_cache
        
        try:
            schema_parts = ['-- MongoDB Database Schema\n']
            
            # Get configured collections
            db_name = Config.MONGO_DB
            collection_names = Config.MONGODB_COLLECTIONS.get(db_name, [])
            
            if not collection_names:
                logger.warning(f"No collections configured for database: {db_name}")
                return "No MongoDB collections configured"
            
            excluded_fields = self._get_excluded_fields()
            
            for collection_name in collection_names:
                if collection_name not in self.db.list_collection_names():
                    logger.warning(f"Collection not found: {collection_name}")
                    continue
                
                collection = self.db[collection_name]
                count = collection.count_documents({})
                
                schema_parts.append(f'Collection: {collection_name}')
                schema_parts.append(f'Document Count: {count}')
                
                # Get sample document
                sample = collection.find_one()
                if sample:
                    # Filter and show fields
                    filtered_fields = [
                        f'"{k}" ({type(v).__name__})'
                        for k, v in sample.items()
                        if k not in excluded_fields
                    ]
                    
                    if filtered_fields:
                        schema_parts.append(f'Fields: {", ".join(filtered_fields[:15])}')
                        if len(filtered_fields) > 15:
                            schema_parts.append(f'... and {len(filtered_fields) - 15} more fields')
                
                schema_parts.append('-' * 60)
            
            self._schema_cache = '\n\n'.join(schema_parts)
            logger.info("MongoDB schema cached")
            return self._schema_cache
            
        except Exception as e:
            logger.error(f"MongoDB schema generation failed: {e}")
            return "Schema generation failed"
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()


# =============================================================================
# DATABASE AGENT
# =============================================================================

postgres_service = PostgresService()
mongo_service = MongoService()
openai_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)


DATABASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_postgres",
            "description": "Query PostgreSQL for products, categories, groups, countries. Use double quotes for identifiers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "SQL SELECT query"
                    }
                },
                "required": ["sql_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_mongodb",
            "description": "Query MongoDB for regulations and standards metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "JSON with collection, action, params"
                    }
                },
                "required": ["operation"]
            }
        }
    }
]


def build_database_agent_prompt() -> str:
    """Minimal prompt for database agent"""
    pg_schema = postgres_service.get_schema()
    mongo_schema = mongo_service.get_schema()

    with open("debug_llm_postgres_schema.txt", "w", encoding="utf-8") as f:
        f.write(pg_schema)
    with open("debug_llm_mongo_schema.txt", "w", encoding="utf-8") as f:
        f.write(mongo_schema)
    return f"""You are a database query specialist.

**PostgreSQL Schema:**
{postgres_service.get_schema()}

**MongoDB Schema:**
{mongo_service.get_schema()}



**Rules:**
1. Use double quotes for PostgreSQL identifiers: "tableName"."columnName"
2. Default LIMIT {Config.DEFAULT_QUERY_LIMIT} for SELECT queries
3. MongoDB operations: {{"collection": "name", "action": "find", "params": {{}}}}
4. Return data in Markdown tables
5. Internal fields are already filtered - focus on user-relevant data

**Response Format:**
- Use Markdown tables for structured data
- Keep summaries concise (2-3 sentences)
- Show only user-relevant fields (code, name, description)"""


async def execute_database_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute database tools"""
    try:
        if tool_name == "query_postgres":
            sql_query = arguments.get("sql_query", "")
            result = postgres_service.execute(sql_query)
            logger.info(f"PostgreSQL tool executed: {sql_query}")
            return json.dumps(result, indent=2, default=str)
        
        elif tool_name == "query_mongodb":
            operation = arguments.get("operation")
            
            try:
                op_dict = json.loads(operation) if isinstance(operation, str) else operation
            except json.JSONDecodeError as e:
                return json.dumps({
                    'success': False,
                    'error': 'Invalid JSON format',
                    'details': str(e)
                })
            
            result = mongo_service.execute(op_dict)
            return json.dumps(result, indent=2, default=str)
        
        else:
            return json.dumps({
                'success': False,
                'error': f'Unknown tool: {tool_name}'
            })
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return json.dumps({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        })


async def query_database_agent(user_query: str, context: str = ""):
    """
    Stream database agent responses
    
    Args:
        user_query: User's database query
        context: Additional context from main agent
    
    Yields:
        Streamed response chunks
    """
    messages = [
        {"role": "system", "content": build_database_agent_prompt()},
        {"role": "user", "content": f"{context}\n\n{user_query}" if context else user_query}
    ]
    
    try:
        stream = await openai_client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=messages,
            tools=DATABASE_TOOLS,
            stream=True
        )
        
        tool_calls = []
        current_tool_call = None
        
        async for chunk in stream:
            delta = chunk.choices[0].delta
            
            # Handle tool calls
            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    if tool_call_delta.index is not None:
                        if tool_call_delta.index >= len(tool_calls):
                            tool_calls.append({
                                "id": tool_call_delta.id,
                                "function": {
                                    "name": tool_call_delta.function.name if tool_call_delta.function else "",
                                    "arguments": ""
                                }
                            })
                        current_tool_call = tool_calls[tool_call_delta.index]
                    
                    if tool_call_delta.function and tool_call_delta.function.arguments:
                        current_tool_call["function"]["arguments"] += tool_call_delta.function.arguments
            
            # Stream text content
            if delta.content:
                yield delta.content
            
            # Check if done
            if chunk.choices[0].finish_reason == "tool_calls":
                # Execute tools
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    
                    result = await execute_database_tool(tool_name, arguments)
                    
                    # Ensure tool_call has type field
                    formatted_tool_call = {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": tool_call["function"]
                    }
                    
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [formatted_tool_call]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })
                
                # Get final response
                final_stream = await openai_client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=messages,
                    stream=True
                )
                
                async for final_chunk in final_stream:
                    if final_chunk.choices[0].delta.content:
                        yield final_chunk.choices[0].delta.content
                
                break
    
    except Exception as e:
        logger.error(f"Database agent error: {e}")
        yield f"\n\nError: {str(e)}"