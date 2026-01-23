"""
Main Agent - Orchestrates file reading, vector search, and database queries
Optimized for low latency with minimal prompt and streaming
"""

import os
import json
import asyncio
import logging
from typing import AsyncGenerator, Dict, Any

from openai import AsyncOpenAI
from dotenv import load_dotenv
# from app.services.vectordb import get_vector_store
# from app.utils.s3_upload import s3_file_reader
from database_agent import query_database_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

KnowledgeBase_COLLECTION_PREFIX = "knowledgebase"
ComplianceWizard_COLLECTION_PREFIX = "complianceWizard"

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    MAX_ITERATIONS = 10


# =============================================================================
# SERVICES
# =============================================================================

openai_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)


# class VectorStoreService:
#     """Vector store operations"""
    
#     async def query(self, collectionID: str, query: str, limit: int = 5) -> Dict[str, Any]:
#         """Query vector store with context formatting"""
#         try:
#             collection_names_list = [
#                 f"{KnowledgeBase_COLLECTION_PREFIX}_{collectionID}",
#                 f"{ComplianceWizard_COLLECTION_PREFIX}_{collectionID}",
#                 "External_Knowledgebase"
#             ]

#             vector_store = get_vector_store()
#             all_docs = []
            
#             for collection_name in collection_names_list:
#                 if vector_store.collection_exist(collection_name):
#                     docs = await asyncio.to_thread(
#                         vector_store.query_hybrid_collection,
#                         collection_name, query, 5
#                     )
#                     if docs:
#                         all_docs.extend(docs)

#             if all_docs:
#                 context = "\n\n".join(d["content"] for d in all_docs)
#                 formatted_prompt = f"""Context from regulatory documents:
# {context}

# User query: {query}

# Provide a concise response (3 lines max) based on the context."""
                
#                 return {'success': True, 'data': formatted_prompt}
#             else:
#                 return {
#                     'success': True, 
#                     'data': 'No relevant regulatory documents found in knowledge base.'
#                 }
            
#         except Exception as e:
#             logger.error(f"Vector store query failed: {e}")
#             return {'success': False, 'error': str(e)}


# vector_service = VectorStoreService()


# =============================================================================
# TOOLS
# =============================================================================

MAIN_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Query Alpha team databases for products, categories, countries, regulations, standards.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language database query"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context (optional)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_vector_store",
            "description": "Search regulatory documents for compliance, requirements, legal interpretations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country_name": {
                        "type": "string",
                        "description": "Country name"
                    },
                    "query": {
                        "type": "string",
                        "description": "Semantic search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["country_name", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_uploaded_file",
            "description": "Read uploaded PDF, DOCX, TXT files. Operations: 'metadata', 'read', 'summary'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Exact filename"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["metadata", "read", "summary"],
                        "default": "read"
                    },
                    "page_start": {
                        "type": "integer",
                        "default": 1
                    },
                    "page_end": {
                        "type": "integer"
                    }
                },
                "required": ["filename"]
            }
        }
    }
]


# =============================================================================
# TOOL EXECUTION
# =============================================================================

async def execute_main_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute main agent tools"""
    try:
        if tool_name == "query_database":
            # Delegate to database agent with streaming
            query = arguments.get("query", "")
            context = arguments.get("context", "")
            
            # Collect streamed response
            response_parts = []
            async for chunk in query_database_agent(query, context):
                response_parts.append(chunk)
            
            return "".join(response_parts)
        
        # elif tool_name == "query_vector_store":
        #     from database_agent import postgres_service
            
        #     country_name = arguments.get("country_name", "")
            
        #     # Get country ID
        #     collection = postgres_service.execute(
        #         f"""SELECT "id" FROM "public"."countries" WHERE "name" = '{country_name}' LIMIT 1;"""
        #     )
            
        #     if collection['count'] == 0:
        #         return json.dumps({
        #             'success': False,
        #             'error': f'Country not found: {country_name}'
        #         })
            
        #     collectionID = collection['data'][0]['id']
        #     query = arguments.get("query", "")
        #     limit = arguments.get("limit", 5)
            
        #     result = await vector_service.query(collectionID, query, limit)
            
        #     if result.get('success'):
        #         return result['data']
        #     else:
        #         return json.dumps(result)
        
        # elif tool_name == "read_uploaded_file":
        #     filename = arguments.get("filename", "")
        #     operation = arguments.get("operation", "read")
        #     page_start = arguments.get("page_start", 1)
        #     page_end = arguments.get("page_end")
            
        #     result = s3_file_reader.read_file(
        #         filename=filename,
        #         operation=operation,
        #         page_start=page_start,
        #         page_end=page_end
        #     )
            
        #     return json.dumps(result, indent=2)
        
        else:
            return json.dumps({'success': False, 'error': f'Unknown tool: {tool_name}'})
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return json.dumps({'success': False, 'error': str(e)})


# =============================================================================
# MAIN AGENT PROMPT (MINIMAL)
# =============================================================================

def build_main_agent_prompt() -> str:
    """Minimal prompt for main orchestration agent"""
    return """You are a helpful assistant with access to databases, files, and regulatory documents.

**Available Tools:**
1. `query_database` - Search products, categories, countries, regulations, standards
2. `query_vector_store` - Search regulatory document content for compliance info
3. `read_uploaded_file` - Read PDF, DOCX, TXT files

**Critical Rules:**
1. **Ask for clarification** if country or data type is missing if not specified (e.g., products, regulations)
2. **Never show** internal IDs, UUIDs, timestamps
3. **Format responses** in Markdown tables
4. **Use tools only** when database/file access is needed
5. **Respond directly** to greetings, math, general questions

**Clarification Examples:**
- User: "Show products" → Ask: "Which country?"
- User: "Give me data about France" → Ask: "What data? (products, regulations, etc.)"

-User: "find the description of the product which has code 123" → Use `query_database` tool

**Tool Usage:**
- Database queries → use `query_database`
- Compliance/legal text → use `query_vector_store`
- Read files → use `read_uploaded_file`

Keep responses concise and user-friendly."""


# =============================================================================
# MAIN AGENT EXECUTION WITH STREAMING
# =============================================================================

async def run_main_agent(user_query: str, conversation_history: list = None) -> AsyncGenerator[str, None]:
    """
    Main agent with streaming support
    
    Args:
        user_query: User's question
        conversation_history: Previous messages (optional)
    
    Yields:
        Streamed response chunks
    """
    messages = [{"role": "system", "content": build_main_agent_prompt()}]
    
    if conversation_history:
        messages.extend(conversation_history)
    
    messages.append({"role": "user", "content": user_query})
    
    iteration = 0
    
    while iteration < Config.MAX_ITERATIONS:
        iteration += 1
        
        try:
            stream = await openai_client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=messages,
                tools=MAIN_AGENT_TOOLS,
                stream=True
            )
            
            tool_calls = []
            current_tool_call = None
            content_buffer = []
            
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
                    content_buffer.append(delta.content)
                    yield delta.content
                
                # Check finish reason
                finish_reason = chunk.choices[0].finish_reason
                
                if finish_reason == "tool_calls":
                    # Execute tools
                    for tool_call in tool_calls:
                        tool_name = tool_call["function"]["name"]
                        arguments = json.loads(tool_call["function"]["arguments"])
                        
                        logger.info(f"Executing tool: {tool_name}")
                        result = await execute_main_tool(tool_name, arguments)
                        
                        # Ensure tool_call has type field
                        formatted_tool_call = {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": tool_call["function"]
                        }
                        
                        # Update conversation
                        messages.append({
                            "role": "assistant",
                            "tool_calls": [formatted_tool_call]
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result
                        })
                    
                    # Continue loop to get final response
                    break
                
                elif finish_reason == "stop":
                    # Conversation complete
                    return
            
            # If we executed tools, continue loop for final response
            if tool_calls:
                continue
            else:
                # No tools needed, done
                return
        
        except Exception as e:
            logger.error(f"Main agent error: {e}")
            yield f"\n\nError: {str(e)}"
            return
    
    yield "\n\n[Maximum iterations reached]"


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example of how to use the main agent"""
    
    # # Example 1: Simple greeting (no tools)
    # print("Example 1: Greeting")
    # async for chunk in run_main_agent("Hello!"):
    #     print(chunk, end="", flush=True)
    # print("\n" + "="*80 + "\n")
    
    # Example 2: Database query
    print("Example 1: Database Query")
    async for chunk in run_main_agent("Show me all products for albania"):
        print(chunk, end="", flush=True)
    print("\n" + "="*80 + "\n")

    print("Example 2: Database Query")
    async for chunk in run_main_agent("Show me all products with code 202511281"):
        print(chunk, end="", flush=True)
    print("\n" + "="*80 + "\n")


    print("Example 3: Database Query")
    async for chunk in run_main_agent("Show me with description Sugar from the Sugarcane"):
        print(chunk, end="", flush=True)
    print("\n" + "="*80 + "\n")


    print("Example 4: Database Query")
    async for chunk in run_main_agent("show me the status of regulation having name as Directive 94/9/EC of the European Parliament and of the Council"):
        print(chunk, end="", flush=True)
    print("\n" + "="*80 + "\n")


    print("Example 5: Database Query")
    async for chunk in run_main_agent("find the regulation metadata of albania"):
        print(chunk, end="", flush=True)
    print("\n" + "="*80 + "\n")



    print("Example 6: Database Query")
    async for chunk in run_main_agent("provide the regulation name and country which has regulation name as Directive 94/9/EC of the European Parliament and of the Council"):
        print(chunk, end="", flush=True)
    print("\n" + "="*80 + "\n")

    print("Example 7: Genearal Query")
    async for chunk in run_main_agent("which database is used in this project"):
        print(chunk, end="", flush=True)
    print("\n" + "="*80 + "\n")
    

    
    # # Example 3: File reading
    # print("Example 3: File Reading")
    # async for chunk in run_main_agent("Read the summary of contract.pdf"):
    #     print(chunk, end="", flush=True)
    # print("\n" + "="*80 + "\n")
    
    # # Example 4: Vector search
    # print("Example 4: Regulatory Search")
    # async for chunk in run_main_agent("What are the safety requirements for cosmetics in France?"):
    #     print(chunk, end="", flush=True)
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(example_usage())