"""
LangGraph Multi-Database Agent with Streaming
Integrates MongoDB, PostgreSQL, and Qdrant vector store
"""

import os
import json
import asyncio
from typing import TypedDict, Annotated, Sequence, Optional, AsyncGenerator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import psycopg2
from psycopg2.extras import RealDictCursor
from pymongo import MongoClient
from bson import json_util
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
POSTGRES_CONFIG = {
    'host': 'localhost',
    'database': 'alpha-product-samir',
    'user': 'postgres',
    'password': 'password',
    'port': 5432
}

MONGO_CONFIG = {
    'connection_string': 'mongodb://root:password@localhost:27020/',
    'database': 'alpha-kcc'
}

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)


# =============================================================================
# DATABASE SERVICES
# =============================================================================

class PostgresService:
    """PostgreSQL database operations"""
    
    def __init__(self):
        self.conn = psycopg2.connect(**POSTGRES_CONFIG)
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        self._schema_cache = None
    
    def execute(self, query: str) -> dict:
        """Execute SQL query"""
        try:
            self.cursor.execute(query)
            if self.cursor.description:
                rows = self.cursor.fetchall()
                return {
                    'success': True,
                    'data': [dict(row) for row in rows],
                    'count': len(rows)
                }
            else:
                self.conn.commit()
                return {
                    'success': True,
                    'message': f'{self.cursor.rowcount} rows affected'
                }
        except Exception as e:
            self.conn.rollback()
            return {'success': False, 'error': str(e)}
    
    def get_schema(self):
        """Export database schema with proper quoted identifiers"""
        if self._schema_cache:
            return self._schema_cache
            
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
            
            self._schema_cache = schema_output
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
    """MongoDB database operations"""
    
    def __init__(self):
        self.client = MongoClient(MONGO_CONFIG['connection_string'])
        self.db = self.client[MONGO_CONFIG['database']]
        self._schema_cache = None
    
    def execute(self, operation: dict) -> dict:
        """Execute MongoDB operation"""
        try:
            collection = self.db[operation['collection']]
            action = operation['action']
            params = operation.get('params', {})
            
            if action == 'find':
                query = params.get('query', {})
                limit = params.get('limit', 10)
                results = list(collection.find(query).limit(limit))
                return {
                    'success': True,
                    'data': json.loads(json_util.dumps(results)),
                    'count': len(results)
                }
            
            elif action == 'countDocuments':
                query = params.get('query', {})
                count = collection.count_documents(query)
                return {'success': True, 'count': count}
            
            elif action == 'aggregate':
                pipeline = params.get('pipeline', [])
                results = list(collection.aggregate(pipeline))
                return {
                    'success': True,
                    'data': json.loads(json_util.dumps(results)),
                    'count': len(results)
                }
            
            else:
                return {'success': False, 'error': f'Unsupported action: {action}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_schema(self) -> str:
        """Get database schema with sample documents"""
        if self._schema_cache:
            return self._schema_cache
            
        try:
            collections = self.db.list_collection_names()
            
            schema_output = '-- MongoDB Database Schema\n\n'
            schema_output += f"Database: {MONGO_CONFIG['database']}\n"
            schema_output += f"Collections: {len(collections)}\n\n"
            
            for collection_name in collections:
                collection = self.db[collection_name]
                count = collection.count_documents({})
                
                schema_output += f"Collection: {collection_name}\n"
                schema_output += f"Document Count: {count}\n"
                
                # Get sample document to show structure
                sample = collection.find_one()
                if sample:
                    schema_output += "Sample Document Structure:\n"
                    schema_output += json_util.dumps(sample, indent=2)
                    schema_output += "\n"
                
                # Get indexes
                indexes = collection.index_information()
                if indexes:
                    schema_output += f"Indexes: {', '.join(indexes.keys())}\n"
                
                schema_output += "\n" + "-" * 60 + "\n\n"
            
            self._schema_cache = schema_output
            return schema_output
        except Exception as e:
            raise Exception(f"Schema export failed: {e}")


class VectorStoreService:
    """Qdrant vector store operations"""
    
    def query(self, collection: str, query: str, limit: int = 5) -> dict:
        """Query vector store"""
        try:
            # TODO: Replace with your actual vector store implementation
            # Example:
            # from your_vector_store import get_vector_store
            # vector_store = get_vector_store()
            # docs = vector_store.query_hybrid_collection(collection, query, limit)
            # return {
            #     'success': True,
            #     'data': [{'content': doc['content']} for doc in docs],
            #     'count': len(docs)
            # }
            
            # Placeholder response
            return {
                'success': True,
                'data': [
                    {'content': f'Document 1 related to: {query}'},
                    {'content': f'Document 2 related to: {query}'}
                ],
                'count': 2
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


# =============================================================================
# LANGCHAIN TOOLS
# =============================================================================

# Initialize services
postgres_service = PostgresService()
mongo_service = MongoService()
vector_service = VectorStoreService()


@tool
def query_postgres(sql_query: str) -> str:
    """
    Query PostgreSQL product database.
    Use for product data, inventory, pricing, and any product-related information.
    
    IMPORTANT: PostgreSQL identifiers are case-sensitive. Always use double quotes for table and column names.
    Example: SELECT "id", "name" FROM "public"."products" WHERE "countryId" = 1
    
    Args:
        sql_query: SQL query to execute (SELECT statements recommended)
    
    Returns:
        JSON string with query results
    """
    result = postgres_service.execute(sql_query)
    return json.dumps(result, indent=2)


@tool
def query_mongodb(operation: str) -> str:
    """
    Query MongoDB regulation database.
    Use for regulations, specifications, compliance data, and regulatory documents.
    
    Args:
        operation: JSON string with format:
            {"collection": "collection_name", "action": "find", "params": {"query": {}, "limit": 10}}
        
        Supported actions:
        - find: {"collection": "name", "action": "find", "params": {"query": {"field": "value"}, "limit": 10}}
        - countDocuments: {"collection": "name", "action": "countDocuments", "params": {"query": {}}}
        - aggregate: {"collection": "name", "action": "aggregate", "params": {"pipeline": [...]}}
    
    Returns:
        JSON string with query results
    """
    try:
        op_dict = json.loads(operation)
        result = mongo_service.execute(op_dict)
        return json.dumps(result, indent=2)
    except json.JSONDecodeError:
        return json.dumps({'success': False, 'error': 'Invalid JSON format'})


@tool
def query_vector_store(collection: str, query: str) -> str:
    """
    Query Qdrant vector store for semantic search.
    Use for finding similar documents, specifications, or related information using semantic similarity.
    
    Args:
        collection: Collection name to search in
        query: Search query text for semantic matching
    
    Returns:
        JSON string with relevant documents
    """
    result = vector_service.query(collection, query)
    return json.dumps(result, indent=2)


# =============================================================================
# LANGGRAPH AGENT
# =============================================================================

class AgentState(TypedDict):
    """Agent state definition"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Create tools list
tools = [query_postgres, query_mongodb, query_vector_store]
tool_node = ToolNode(tools)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end"""
    last_message = state["messages"][-1]
    
    # If there are no tool calls, we're done
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    
    return "continue"


async def call_model(state: AgentState) -> dict:
    """Call the LLM with context"""
    
    # Build system message with schemas
    system_message = f"""You are a helpful database assistant with access to multiple databases.

<role>
You help users interact with databases by generating accurate queries and providing clear, concise responses.
</role>

<databases>

1. PostgreSQL (Product Database):
{postgres_service.get_schema()}

2. MongoDB (Regulations Database):
{mongo_service.get_schema()}

3. Qdrant Vector Store:
   - For semantic search across specifications and documents
   - Use when users need to find similar or related content

</databases>

<rules>
    <rule>Analyze the user query carefully to determine if database access is needed</rule>
    <rule>Use tools ONLY when specific data retrieval from databases is required</rule>
    <rule>For general questions or greetings, answer directly without using tools</rule>
    <rule>PostgreSQL identifiers MUST use double quotes for case-sensitive names (e.g., "countryId", not countryId)</rule>
    <rule>Use the exact table and column names as shown in the schema</rule>
    <rule>Default to LIMIT 10 for SELECT queries unless user specifies otherwise</rule>
    <rule>Keep final responses concise and clear (3-4 sentences)</rule>
    <rule>Format query results in a readable way</rule>
    <rule>Always double-check SQL syntax before generating queries</rule>
</rules>

<tool_usage_guide>
    - query_postgres: Use for product data, inventory, pricing, country information
    - query_mongodb: Use for regulations, specifications, compliance documents
    - query_vector_store: Use for semantic search when looking for similar or related documents
</tool_usage_guide>

<examples>
Example 1 - PostgreSQL Query:
User: "How many products are in Albania?"
Tool: query_postgres
SQL: SELECT COUNT(*) AS "product_count" FROM "public"."products" WHERE "countryId" = (SELECT "id" FROM "public"."countries" WHERE "name" = 'Albania')

Example 2 - MongoDB Query:
User: "Find regulations about safety"
Tool: query_mongodb
Operation: {{"collection": "regulations", "action": "find", "params": {{"query": {{"topic": "safety"}}, "limit": 10}}}}

Example 3 - No Tool Needed:
User: "What is 2+2?"
Response: "2+2 equals 4. This is a simple arithmetic calculation."
</examples>
"""
    
    messages = [{"role": "system", "content": system_message}] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    
    return {"messages": [response]}


# Build graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

# Add edge from tools back to agent
workflow.add_edge("tools", "agent")

# Compile graph
graph = workflow.compile()


# =============================================================================
# STREAMING INTERFACE
# =============================================================================

async def process_query_stream(
    user_query: str,
    collection_name: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Process user query with streaming response.
    
    Args:
        user_query: User's question/query
        collection_name: Optional vector store collection for context
    
    Yields:
        SSE formatted chunks
    """
    
    try:
        # Add vector context if collection provided
        enhanced_query = user_query
        if collection_name:
            vector_result = vector_service.query(collection_name, user_query)
            if vector_result['success'] and vector_result['data']:
                context = "\n\n".join(d['content'] for d in vector_result['data'])
                enhanced_query = f"""Context from vector store:
{context}

User Query: {user_query}

Use the context if relevant to answer the query."""
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=enhanced_query)]
        }
        
        # Stream through graph
        async for event in graph.astream(initial_state, stream_mode="values"):
            last_message = event["messages"][-1]
            
            # Stream AI messages
            if isinstance(last_message, AIMessage):
                if last_message.content:
                    # Stream content tokens
                    for char in last_message.content:
                        yield f"data: {json.dumps({'type': 'content', 'data': char})}\n\n"
                        await asyncio.sleep(0.01)  # Throttle streaming
                
                # Handle tool calls
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        tool_info = {
                            'type': 'tool_call',
                            'tool': tool_call['name'],
                            'args': tool_call['args']
                        }
                        yield f"data: {json.dumps(tool_info)}\n\n"
            
            # Stream tool results
            elif isinstance(last_message, ToolMessage):
                tool_result = {
                    'type': 'tool_result',
                    'content': last_message.content[:200] + '...'  # Truncate for streaming
                }
                yield f"data: {json.dumps(tool_result)}\n\n"
        
        # Send completion
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
    
    except Exception as e:
        error_data = {'type': 'error', 'message': str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

async def main():
    """Example usage"""
    
    print("=== LangGraph Multi-Database Agent ===\n")
    
    # Test queries
    queries = [
        "How many products are in Albania?",
        "Find regulations in MongoDB",
        "What is the capital of France?",  # No tool needed
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        print("💬 Response: ", end="", flush=True)
        
        async for chunk in process_query_stream(query):
            if chunk.startswith("data: "):
                data = json.loads(chunk[6:])
                
                if data['type'] == 'content':
                    print(data['data'], end="", flush=True)
                elif data['type'] == 'tool_call':
                    print(f"\n🔧 Using tool: {data['tool']}")
                    print(f"   Args: {json.dumps(data['args'], indent=6)}")
                elif data['type'] == 'complete':
                    print("\n✅ Complete\n")


if __name__ == "__main__":
    asyncio.run(main())