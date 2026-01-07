"""
Conversation History Manager for Multi-Database Agent
Maintains context across multiple queries
"""

import json
from typing import Optional, AsyncGenerator, Dict, Any, List
from datetime import datetime
from collections import deque
import asyncio
import logging

logger = logging.getLogger(__name__)

class ConversationHistory:
    """Manages conversation history with token limit awareness"""
    
    def __init__(self, max_messages: int = 20, max_tokens: int = 4000):
        """
        Initialize conversation history
        
        Args:
            max_messages: Maximum number of message pairs to keep
            max_tokens: Approximate max tokens for history (rough estimate)
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.messages: deque = deque(maxlen=max_messages * 4)  # Allow for tool calls
        self.system_prompt: Optional[str] = None
        self.metadata: Dict[str, Any] = {
            'created_at': datetime.utcnow().isoformat(),
            'total_queries': 0
        }
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt"""
        self.system_prompt = prompt
    
    def add_user_message(self, content: str):
        """Add user message to history"""
        self.messages.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.metadata['total_queries'] += 1
    
    def add_assistant_message(self, content: str, tool_calls: Optional[List[Dict]] = None):
        """Add assistant message to history"""
        message = {
            "role": "assistant",
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        
        self.messages.append(message)
    
    def add_tool_message(self, tool_call_id: str, name: str, content: str):
        """Add tool result to history"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_messages_for_api(self) -> List[Dict[str, Any]]:
        """
        Get messages formatted for OpenAI API
        Returns system prompt + conversation history
        """
        api_messages = []
        
        # Add system prompt
        if self.system_prompt:
            api_messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        # Add conversation history (without timestamps)
        for msg in self.messages:
            api_msg = {
                "role": msg["role"],
                "content": msg.get("content", "")
            }
            
            # Add tool_calls if present
            if "tool_calls" in msg:
                api_msg["tool_calls"] = msg["tool_calls"]
            
            # Add tool-specific fields
            if msg["role"] == "tool":
                api_msg["tool_call_id"] = msg["tool_call_id"]
                api_msg["name"] = msg["name"]
            
            api_messages.append(api_msg)
        
        return api_messages
    
    def get_summary(self) -> Dict[str, Any]:
        """Get conversation summary"""
        user_messages = sum(1 for m in self.messages if m["role"] == "user")
        assistant_messages = sum(1 for m in self.messages if m["role"] == "assistant")
        tool_messages = sum(1 for m in self.messages if m["role"] == "tool")
        
        return {
            "total_messages": len(self.messages),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "tool_messages": tool_messages,
            "total_queries": self.metadata['total_queries'],
            "created_at": self.metadata['created_at']
        }
    
    def clear(self):
        """Clear conversation history"""
        self.messages.clear()
        self.metadata['total_queries'] = 0
        self.metadata['created_at'] = datetime.utcnow().isoformat()
    
    def export_history(self) -> str:
        """Export history as JSON string"""
        return json.dumps({
            "system_prompt": self.system_prompt,
            "messages": list(self.messages),
            "metadata": self.metadata
        }, indent=2, default=str)
    
    def import_history(self, history_json: str):
        """Import history from JSON string"""
        data = json.loads(history_json)
        self.system_prompt = data.get("system_prompt")
        self.messages = deque(data.get("messages", []), maxlen=self.max_messages * 4)
        self.metadata = data.get("metadata", self.metadata)


class ConversationManager:
    """Manages multiple conversation sessions"""
    
    def __init__(self):
        self.conversations: Dict[str, ConversationHistory] = {}
    
    def create_conversation(
        self, 
        conversation_id: str, 
        system_prompt: str,
        max_messages: int = 20
    ) -> ConversationHistory:
        """Create a new conversation"""
        conversation = ConversationHistory(max_messages=max_messages)
        conversation.set_system_prompt(system_prompt)
        self.conversations[conversation_id] = conversation
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        """Get existing conversation"""
        return self.conversations.get(conversation_id)
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations with summaries"""
        return [
            {
                "id": conv_id,
                "summary": conv.get_summary()
            }
            for conv_id, conv in self.conversations.items()
        ]


# =============================================================================
# UPDATED AGENT FUNCTION WITH HISTORY
# =============================================================================

async def process_query_stream_with_history(
    user_query: str,
    conversation: ConversationHistory,
    collection_name: Optional[str] = None,
    openai_client=None,
    vector_service=None,
    execute_tool=None,
    TOOLS=None,
    Config=None
) -> AsyncGenerator[str, None]:
    """
    Process user query with streaming response and conversation history
    
    Args:
        user_query: User's question
        conversation: ConversationHistory object
        collection_name: Optional vector store collection for context
        openai_client: OpenAI client instance
        vector_service: Vector store service instance
        execute_tool: Tool execution function
        TOOLS: Tool definitions
        Config: Configuration object
        
    Yields:
        SSE formatted data chunks
    """
    try:
        logger.info(f"Processing query with history: {user_query}")
        logger.info(f"Current conversation has {len(conversation.messages)} messages")
        
        # Add user message to history
        conversation.add_user_message(user_query)
        logger.info(f"[History] After adding user, messages={len(conversation.messages)}")

        # Get conversation history for API (includes all previous context)
        messages = conversation.get_messages_for_api()
        
        logger.info(f"Sending {len(messages)} messages to API (including system prompt)")
        for i, msg in enumerate(messages):
            logger.info(f"[History] msg[{i}] role={msg['role']} content={str(msg.get('content'))[:150]}")
        # Add vector context if provided (only to current query, not history)
        if collection_name and vector_service:
            logger.info(f"Adding vector context from: {collection_name}")
            vector_result = vector_service.query(collection_name, user_query, limit=3)
            
            if vector_result['success'] and vector_result['data']:
                context_docs = [doc['content'] for doc in vector_result['data']]
                context = '\n\n'.join(context_docs)
                
                # Add context as a separate system message (temporary, not saved to history)
                context_message = {
                    "role": "system",
                    "content": f"""<context>
{context}
</context>

Use the context above if relevant to answer the current user query."""
                }
                # Insert before the last user message
                messages.insert(-1, context_message)
        
        iteration = 0
        full_assistant_content = ""
        all_tool_calls = []
        
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
            current_content = ""
            tool_calls = []
            
            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                
                if not delta:
                    continue
                
                # Handle tool calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if tc_delta.index is not None:
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
                    current_content += delta.content
                    
                    # Emit content character by character
                    for char in delta.content:
                        content_data = {'type': 'content', 'data': char}
                        yield f"data: {json.dumps(content_data)}\n\n"
                        await asyncio.sleep(Config.STREAMING_DELAY)
            
            # Accumulate content for history
            full_assistant_content += current_content
            
            # Add assistant message to conversation messages FOR NEXT ITERATION
            assistant_message = {"role": "assistant", "content": current_content}
            
            # If there are tool calls, process them
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)
                all_tool_calls.extend(tool_calls)
                
                # Emit tool calls
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
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": tool_result
                    }
                    messages.append(tool_msg)
                    
                    # Add to conversation history immediately
                    conversation.add_tool_message(
                        tool_call["id"],
                        function_name,
                        tool_result
                    )
                    
                    # Emit tool result (truncated for display)
                    truncated_result = tool_result[:500] + '...' if len(tool_result) > 500 else tool_result
                    tool_result_data = {
                        'type': 'tool_result',
                        'tool': function_name,
                        'content': truncated_result,
                        'tool_call_id': tool_call["id"]
                    }
                    yield f"data: {json.dumps(tool_result_data)}\n\n"
                
                # Continue loop to get final response
                continue
            
            else:
                # No tool calls, we have final response
                messages.append(assistant_message)
                break
        
        # Add complete assistant response to history (only once at the end)
        conversation.add_assistant_message(
            full_assistant_content, 
            all_tool_calls if all_tool_calls else None
        )
        
        logger.info(f"Conversation now has {len(conversation.messages)} messages")
        
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


async def process_query_with_history(
    user_query: str,
    conversation: ConversationHistory,
    collection_name: Optional[str] = None,
    openai_client=None,
    vector_service=None,
    execute_tool=None,
    TOOLS=None,
    Config=None
) -> Dict[str, Any]:
    """
    Process query with conversation history (non-streaming)
    
    Args:
        user_query: User's question
        conversation: ConversationHistory object
        collection_name: Optional vector store collection
        openai_client: OpenAI client instance
        vector_service: Vector store service instance
        execute_tool: Tool execution function
        TOOLS: Tool definitions
        Config: Configuration object
        
    Returns:
        Complete response dictionary
    """
    try:
        logger.info(f"Processing query with history (non-streaming): {user_query}")
        
        # Add user message to history
        conversation.add_user_message(user_query)
        
        # Get conversation history
        messages = conversation.get_messages_for_api()
        
        # Add vector context if provided
        if collection_name and vector_service:
            vector_result = vector_service.query(collection_name, user_query, limit=3)
            if vector_result['success'] and vector_result['data']:
                context = '\n\n'.join(doc['content'] for doc in vector_result['data'])
                context_message = {
                    "role": "system",
                    "content": f"""<context>
{context}
</context>

Use the context if relevant."""
                }
                messages.insert(-1, context_message)
        
        all_tool_calls = []
        iteration = 0
        final_content = ""
        
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
            
            final_content = assistant_message.content or ""
            
            # Check for tool calls
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    all_tool_calls.append({
                        'tool': function_name,
                        'args': function_args
                    })
                    
                    # Execute tool
                    tool_result = execute_tool(function_name, function_args)
                    
                    # Add tool result to messages
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result
                    }
                    messages.append(tool_msg)
                    
                    # Add to conversation history
                    conversation.add_tool_message(
                        tool_call.id,
                        function_name,
                        tool_result
                    )
                
                # Continue loop
                continue
            
            else:
                # No more tool calls, we have final response
                break
        
        # Add assistant response to history
        conversation.add_assistant_message(final_content, all_tool_calls if all_tool_calls else None)
        
        result = {
            'success': True,
            'response': final_content,
            'tool_calls': all_tool_calls,
            'conversation_summary': conversation.get_summary(),
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