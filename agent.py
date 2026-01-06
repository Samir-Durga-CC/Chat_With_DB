"""
Multi-Database Agent with OpenAI (without LangGraph)
Direct OpenAI API implementation with streaming support
"""

import os
import json
import asyncio
import logging
from typing import Optional, AsyncGenerator, Dict, Any, List
from datetime import datetime

from openai import AsyncOpenAI
import psycopg2
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient
from bson import json_util
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration"""
    
    # PostgreSQL
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'alpha-product-samir')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
    
    # MongoDB
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://root:password@localhost:27020/')
    MONGO_DB = os.getenv('MONGO_DB', 'alpha-kcc')
    
    # OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    
    # Agent settings
    MAX_ITERATIONS = int(os.getenv('MAX_ITERATIONS', 10))
    STREAMING_DELAY = float(os.getenv('STREAMING_DELAY', 0.01))
    DEFAULT_QUERY_LIMIT = int(os.getenv('DEFAULT_QUERY_LIMIT', 10))


# =============================================================================
# DATABASE SERVICES
# =============================================================================

class PostgresService:
    """PostgreSQL database operations"""
    
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
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("PostgreSQL connection established")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Execute SQL query with error handling"""
        try:
            if self.conn.closed:
                self._connect()
            
            self.cursor.execute(query)
            
            if self.cursor.description:
                rows = self.cursor.fetchall()
                data = [dict(row) for row in rows]
                logger.info(f"Query executed successfully. Rows returned: {len(data)}")
                return {
                    'success': True,
                    'data': data,
                    'count': len(data),
                    'query': query
                }
            else:
                self.conn.commit()
                affected_rows = self.cursor.rowcount
                logger.info(f"Query executed successfully. Rows affected: {affected_rows}")
                return {
                    'success': True,
                    'message': f'{affected_rows} rows affected',
                    'affected_rows': affected_rows,
                    'query': query
                }
                
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Query execution failed: {e}\nQuery: {query}")
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'query': query
            }
    
    def get_schema(self) -> str:
        """Get database schema"""
        if self._schema_cache:
            return self._schema_cache
        
        schema_query = """
        SELECT 
            t.schemaname as schema_name,
            t.tablename as table_name,
            array_agg(
                '"' || c.column_name || '" ' || c.udt_name ||
                CASE 
                    WHEN c.character_maximum_length IS NOT NULL 
                    THEN '(' || c.character_maximum_length || ')'
                    ELSE ''
                END ||
                CASE 
                    WHEN c.is_nullable = 'NO' THEN ' NOT NULL'
                    ELSE ''
                END
                ORDER BY c.ordinal_position
            ) as columns
        FROM pg_tables t
        JOIN information_schema.columns c 
            ON t.schemaname = c.table_schema 
            AND t.tablename = c.table_name
        WHERE t.schemaname NOT IN ('pg_catalog', 'information_schema')
        GROUP BY t.schemaname, t.tablename
        ORDER BY t.schemaname, t.tablename;
        """
        
        try:
            self.cursor.execute(schema_query)
            results = self.cursor.fetchall()
            
            schema_parts = ['-- PostgreSQL Database Schema\n']
            
            for row in results:
                schema_name = row['schema_name']
                table_name = row['table_name']
                columns = ', '.join(row['columns'])
                
                schema_parts.append(
                    f'CREATE TABLE "{schema_name}"."{table_name}" ({columns});'
                )
            
            self._schema_cache = '\n\n'.join(schema_parts)
            logger.info("Schema cached successfully")
            return self._schema_cache
            
        except Exception as e:
            logger.error(f"Schema export failed: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("PostgreSQL connection closed")


class MongoService:
    """MongoDB database operations"""
    
    def __init__(self):
        self.client = MongoClient(Config.MONGO_URI)
        self.db = self.client[Config.MONGO_DB]
        self._schema_cache = None
        logger.info("MongoDB connection established")
    
    def execute(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MongoDB operation"""
        try:
            
            logger.info(f"MongoDB operation: {operation}")
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
                
                logger.info(f"MongoDB find: {len(results)} documents returned")
                return {
                    'success': True,
                    'data': json.loads(json_util.dumps(results)),
                    'count': len(results),
                    'operation': operation
                }
            
            elif action == 'countDocuments':
                query = params.get('query', {})
                count = collection.count_documents(query)
                
                logger.info(f"MongoDB count: {count} documents")
                return {
                    'success': True,
                    'count': count,
                    'operation': operation
                }
            
            elif action == 'aggregate':
                pipeline = params.get('pipeline', [])
                results = list(collection.aggregate(pipeline))
                
                logger.info(f"MongoDB aggregate: {len(results)} documents returned")
                return {
                    'success': True,
                    'data': json.loads(json_util.dumps(results)),
                    'count': len(results),
                    'operation': operation
                }
            
            elif action == 'findOne':
                query = params.get('query', {})
                projection = params.get('projection')
                result = collection.find_one(query, projection)
                
                logger.info(f"MongoDB findOne: {'Found' if result else 'Not found'}")
                return {
                    'success': True,
                    'data': json.loads(json_util.dumps(result)) if result else None,
                    'operation': operation
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
                'error_type': type(e).__name__,
                'operation': operation
            }
    
    def get_schema(self) -> str:
        """Get MongoDB schema"""
        if self._schema_cache:
            return self._schema_cache
        
        try:
            collections = self.db.list_collection_names()
            
            schema_parts = [
                '-- MongoDB Database Schema\n',
                f'Database: {Config.MONGO_DB}',
                f'Collections: {len(collections)}\n'
            ]
            
            for collection_name in collections:
                collection = self.db[collection_name]
                count = collection.count_documents({})
                
                schema_parts.append(f'Collection: {collection_name}')
                schema_parts.append(f'Document Count: {count}')
                
                sample = collection.find_one()
                if sample:
                    schema_parts.append('Sample Document:')
                    schema_parts.append(json_util.dumps(sample, indent=2))
                
                indexes = collection.index_information()
                if indexes:
                    schema_parts.append(f"Indexes: {', '.join(indexes.keys())}")
                
                schema_parts.append('-' * 60)
            
            self._schema_cache = '\n\n'.join(schema_parts)
            logger.info("MongoDB schema cached successfully")
            return self._schema_cache
            
        except Exception as e:
            logger.error(f"MongoDB schema export failed: {e}")
            raise
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        logger.info("MongoDB connection closed")


class VectorStoreService:
    """Qdrant vector store operations (placeholder)"""
    
    def query(self, collection: str, query: str, limit: int = 5) -> Dict[str, Any]:
        """Query vector store"""
        try:
            logger.info(f"Vector search: {query} (limit: {limit})")
            
            return {
                'success': True,
                'data': [
                    {'content': f'Document 1 related to: {query}', 'score': 0.95},
                    {'content': f'Document 2 related to: {query}', 'score': 0.87}
                ],
                'count': 2,
                'collection': collection,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Vector store query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }


# =============================================================================
# INITIALIZE SERVICES
# =============================================================================

postgres_service = PostgresService()
mongo_service = MongoService()
vector_service = VectorStoreService()
openai_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_postgres",
            "description": """Query PostgreSQL product database.
            
Use this tool for:
- Product information (name, code, description, pricing)
- Inventory and stock data
- Country and location data

CRITICAL: PostgreSQL identifiers are case-sensitive!
Always use double quotes for table and column names.

Examples:
- SELECT "id", "name" FROM "public"."products" WHERE "code" = '123'
- SELECT COUNT(*) FROM "public"."products" WHERE "countryId" = 1""",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "SQL SELECT query (read-only recommended)"
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
            "description": """Query MongoDB regulation database.
            
Use this tool for:
- Regulations and compliance data
- Specifications and standards
- Regulatory documents

Operation format:
{
    "collection": "collection_name",
    "action": "find|findOne|countDocuments|aggregate",
    "params": {
        "query": {"field": "value"},
        "limit": 10,
        "projection": {"field": 1}
    }
}""",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "JSON string with collection, action, and params"
                    }
                },
                "required": ["operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_vector_store",
            "description": """Query Qdrant vector store for semantic search.
            
Use this tool for:
- Finding similar documents
- Semantic search across specifications
- Related content discovery""",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Vector store collection name"
                    },
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    }
                },
                "required": ["collection", "query"]
            }
        }
    }
]


# =============================================================================
# TOOL EXECUTION
# =============================================================================

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool and return results"""
    try:
        if tool_name == "query_postgres":
            sql_query = arguments.get("sql_query", "")
            result = postgres_service.execute(sql_query)
            return json.dumps(result, indent=2,default=str)
        
        elif tool_name == "query_mongodb":
            # Try operation string first (preferred)
            operation = arguments.get("operation")
            
            # If no operation string, try direct object format
            if not operation:
                # Direct object from model
                op_dict = {
                    "collection": arguments.get("collection"),
                    "action": arguments.get("action"),
                    "params": arguments.get("params", {})
                }
            else:
                try:
                    op_dict = json.loads(operation)
                except json.JSONDecodeError as e:
                    return json.dumps({
                        'success': False,
                        'error': 'Invalid JSON format',
                        'details': str(e),
                        'received': operation[:100] + '...'
                    }, indent=2, default=str)
            
            # Validate required fields
            if not op_dict.get('collection'):
                return json.dumps({
                    'success': False,
                    'error': 'Collection name required',
                    'received': op_dict
                }, indent=2, default=str)
            
            result = mongo_service.execute(op_dict)
            logger.info(f"MongoDB operation executed: {result}")
            return json.dumps(result, indent=2, default=str)

        
        elif tool_name == "query_vector_store":
            collection = arguments.get("collection", "")
            query = arguments.get("query", "")
            limit = arguments.get("limit", 5)
            result = vector_service.query(collection, query, limit)
            return json.dumps(result, indent=2,default=str)
        
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


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

def build_system_prompt() -> str:
    """Build system prompt"""
    return f"""You are an intelligent database assistant with access to multiple databases.

<role>
You help users query and analyze data across PostgreSQL, MongoDB, and vector stores.
You provide accurate, concise, and well-formatted responses.
</role>

<capabilities>
1. PostgreSQL Database (Product Data):
{postgres_service.get_schema()}

2. MongoDB Database (Regulations):
{mongo_service.get_schema()}

3. Qdrant Vector Store (Semantic Search):
   - For finding similar documents and specifications
   - Use for context-based queries
</capabilities>

<critical_rules>
1. **Tool Usage Intelligence**:
   - Use tools ONLY when database access is required
   - For greetings, general questions, or math: respond directly
   - Analyze the query intent before deciding to use tools

2. **PostgreSQL Requirements**:
   - ALWAYS use double quotes for identifiers: "tableName", "columnName"
   - Use exact names from schema (case-sensitive)
   - Default to LIMIT {Config.DEFAULT_QUERY_LIMIT} for SELECT queries
   - Validate SQL syntax before execution

3. **MongoDB Operations**:
   - Always specify collection name
   - Use proper JSON format for operations
   - Include query filters and limits

4. **Response Format**:
   - After receiving tool results, provide a clear summary
   - Format data in readable tables or lists
   - Keep responses concise (3-5 sentences for summaries)
   - Always acknowledge what was found

5. **Error Handling**:
   - If query fails, explain the error clearly
   - Suggest corrections for syntax errors
   - Don't retry failed queries without modification

6. **Data privacy and simplification**:
   - Treat columns like "id", "createdAt", "updatedAt", "countryId", "categoryId",
     "productMetaId" and other values as internal identifiers.
   - Do NOT show these raw values to the user.
   - Instead, summarize products in human language and only include:
     - product code
     - user‑relevant descriptive fields (e.g., name, description, synonyms)
     - derived info such as country name if you can look it up.
   - Never include UUIDs, database IDs, or raw timestamps in the user‑facing answer.

</critical_rules>

<examples>
Example 1 - Direct Answer:
User: "What is 2+2?"
Assistant: "2+2 equals 4."

Example 2 - PostgreSQL Query:
User: "Show me product with code 123"
Tool: query_postgres
SQL: SELECT * FROM "public"."products" WHERE "code" = '123' LIMIT 10
Response: "I found the product with code 123. It is [name], priced at [price], available in [country]."

Example 3 - MongoDB Query:
User: "Find safety regulations"
Tool: query_mongodb
Operation: {{"collection": "regulations", "action": "find", "params": {{"query": {{"topic": "safety"}}, "limit": 5}}}}
Response: "I found 5 safety regulations. Here are the key findings: [summary]"


</examples>

<response_guidelines>
- Be conversational and helpful
- Provide context for your answers
- Format data clearly (use tables/lists when appropriate)
- Always summarize tool results in natural language
- If no results found, say so clearly and suggest alternatives
</response_guidelines>
"""


# =============================================================================
# AGENT WITH STREAMING
# =============================================================================

async def process_query_stream(
    user_query: str,
    collection_name: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Process user query with streaming response using OpenAI
    
    Args:
        user_query: User's question
        collection_name: Optional vector store collection for context
        
    Yields:
        SSE formatted data chunks
    """
    try:
        logger.info(f"Processing query: {user_query}")
        
        # Add vector context if provided
        enhanced_query = user_query
        if collection_name:
            logger.info(f"Adding vector context from: {collection_name}")
            vector_result = vector_service.query(collection_name, user_query, limit=3)
            
            if vector_result['success'] and vector_result['data']:
                context_docs = [doc['content'] for doc in vector_result['data']]
                context = '\n\n'.join(context_docs)
                
                enhanced_query = f"""<context>
{context}
</context>

<user_query>
{user_query}
</user_query>

Use the context above if relevant to answer the user's query."""
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": enhanced_query}
        ]
        
        iteration = 0
        
        while iteration < Config.MAX_ITERATIONS:
            iteration += 1
            logger.info(f"Iteration {iteration}/{Config.MAX_ITERATIONS}")
            
            # Call OpenAI with streaming
            response = await openai_client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=True
            )
            
            # Collect streaming response
            full_content = ""
            tool_calls = []
            current_tool_call = None
            
            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                
                if not delta:
                    continue
                
                # Handle tool calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if tc_delta.index is not None:
                            # Ensure we have a tool call at this index
                            while len(tool_calls) <= tc_delta.index:
                                tool_calls.append({
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            current_tool_call = tool_calls[tc_delta.index]
                            
                            if tc_delta.id:
                                current_tool_call["id"] = tc_delta.id
                            
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    current_tool_call["function"]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    current_tool_call["function"]["arguments"] += tc_delta.function.arguments
                
                # Handle content
                if delta.content:
                    full_content += delta.content
                    
                    # Emit content character by character
                    for char in delta.content:
                        content_data = {'type': 'content', 'data': char}
                        yield f"data: {json.dumps(content_data)}\n\n"
                        await asyncio.sleep(Config.STREAMING_DELAY)
            
            # Add assistant message to history
            assistant_message = {"role": "assistant", "content": full_content}
            
            # If there are tool calls, process them
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)
                
                # Emit tool calls
                logger.info(f"Emitting {len(tool_calls)} tool calls")
                logger.info(f"Tool calls details: {tool_calls}")
                for tool_call in tool_calls:
                    tool_data = {
                        'type': 'tool_call',
                        'tool': tool_call["function"]["name"],
                        'args': json.loads(tool_call["function"]["arguments"]),
                        'id': tool_call["id"]
                    }
                    yield f"data: {json.dumps(tool_data)}\n\n"
                    logger.info(f"Emitted tool call: {tool_call['function']['name']}")
                
                # Execute tools
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = json.loads(tool_call["function"]["arguments"])
                    
                    logger.info(f"Executing tool: {function_name}")
                    tool_result = execute_tool(function_name, function_args)
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": tool_result
                    })
                    
                    # Emit tool result (truncated)
                    truncated_result = tool_result[:500] + '...' if len(tool_result) > 500 else tool_result
                    tool_result_data = {
                        'type': 'tool_result',
                        'tool': function_name,
                        'content': truncated_result,
                        'tool_call_id': tool_call["id"]
                    }
                    logger.info(f"Tool result emitted:=========== {tool_result_data}")
                    yield f"data: {json.dumps(tool_result_data)}\n\n"
                
                # Continue loop to get final response
                continue
            
            else:
                # No tool calls, we have final response
                messages.append(assistant_message)
                break
        
        # Send completion
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        logger.info("Query processing complete")
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        error_data = {
            'type': 'error',
            'message': str(e),
            'error_type': type(e).__name__
        }
        yield f"data: {json.dumps(error_data)}\n\n"


async def process_query(
    user_query: str,
    collection_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process query and return complete response (non-streaming)
    
    Args:
        user_query: User's question
        collection_name: Optional vector store collection
        
    Returns:
        Complete response dictionary
    """
    try:
        logger.info(f"Processing query (non-streaming): {user_query}")
        
        # Add vector context if provided
        enhanced_query = user_query
        if collection_name:
            vector_result = vector_service.query(collection_name, user_query, limit=3)
            if vector_result['success'] and vector_result['data']:
                context = '\n\n'.join(doc['content'] for doc in vector_result['data'])
                enhanced_query = f"""<context>
{context}
</context>

<user_query>
{user_query}
</user_query>

Use the context if relevant."""
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": enhanced_query}
        ]
        
        all_tool_calls = []
        iteration = 0
        
        while iteration < Config.MAX_ITERATIONS:
            iteration += 1
            
            # Call OpenAI
            response = await openai_client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            messages.append(assistant_message)
            
            # Check for tool calls
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    all_tool_calls.append({
                        'type': 'tool_call',
                        'tool': function_name,
                        'args': function_args
                    })
                    
                    # Execute tool
                    tool_result = execute_tool(function_name, function_args)
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result
                    })
                
                # Continue loop
                continue
            
            else:
                # No more tool calls, we have final response
                break
        
        result = {
            'success': True,
            'response': assistant_message.content or "",
            'tool_calls': all_tool_calls,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info("Query processing complete")
        return result
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'timestamp': datetime.utcnow().isoformat()
        }


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

async def example_streaming():
    """Example of streaming usage"""
    
    print("=== Multi-Database Agent with OpenAI (Streaming) ===\n")
    
    queries = [
        "Show me product with code 123",
        "How many products are in the database?",
        "What is 2+2?",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        print("Response: ", end="", flush=True)
        
        async for chunk in process_query_stream(query):
            if chunk.startswith("data: "):
                try:
                    data = json.loads(chunk[6:])
                    
                    if data['type'] == 'content':
                        print(data['data'], end="", flush=True)
                    
                    elif data['type'] == 'tool_call':
                        print(f"\n\nTool: {data['tool']}")
                        print(f"Args: {json.dumps(data['args'], indent=2)}")
                        print("\nResponse: ", end="", flush=True)
                    
                    elif data['type'] == 'complete':
                        print("\n\nComplete")
                    
                    elif data['type'] == 'error':
                        print(f"\n\nError: {data['message']}")
                
                except json.JSONDecodeError:
                    pass
        
        await asyncio.sleep(1)


async def example_non_streaming():
    """Example of non-streaming usage"""
    
    print("\n\n=== Multi-Database Agent with OpenAI (Non-Streaming) ===\n")
    
    query = "Show me product with code 123"
    print(f"Query: {query}\n")
    
    result = await process_query(query)
    
    print("Full Response:")
    print(json.dumps(result, indent=2))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main execution function"""
    try:
        # Run streaming example
        await example_streaming()
        
        # Run non-streaming example
        await example_non_streaming()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Main execution error: {e}", exc_info=True)
    finally:
        # Cleanup
        print("\n\nCleaning up...")
        postgres_service.close()
        mongo_service.close()
        print("Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())