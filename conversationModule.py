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
    """
    Optimized conversation store.
    Stores ONLY user/assistant message pairs (no tools).
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: deque = deque(maxlen=max_messages * 2)  # user + assistant
        self.system_prompt: Optional[str] = None
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(),
            "total_queries": 0,
        }

    def set_system_prompt(self, prompt: str):
        """Set or update the system prompt."""
        self.system_prompt = prompt

    def add_user_message(self, content: str):
        """Add a user message (PERSISTED)."""
        self.messages.append(
            {
                "role": "user",
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self.metadata["total_queries"] += 1

    def add_assistant_message(self, content: str):
        """Add ONLY the final assistant message (NO tool_calls)."""
        self.messages.append(
            {
                "role": "assistant",
                "content": content,  # Final summarized response
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def get_messages_for_api(self) -> List[Dict[str, Any]]:
        """
        Return ONLY user/assistant pairs.
        Safe for OpenAI API usage (no tools, no validation issues).
        """
        api_messages: List[Dict[str, Any]] = []

        if self.system_prompt:
            api_messages.append(
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            )

        # Safe: No tool_calls or tool_call_id fields
        for msg in self.messages:
            api_messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                }
            )

        logger.info(f"[History] Clean {len(api_messages)} messages for OpenAI")
        return api_messages

    def get_summary(self) -> Dict[str, Any]:
        """Return a lightweight summary of the conversation."""
        user_msgs = sum(1 for m in self.messages if m["role"] == "user")

        return {
            "total_messages": len(self.messages),
            "user_messages": user_msgs,
            "assistant_messages": len(self.messages) - user_msgs,
            "total_queries": self.metadata["total_queries"],
        }

    def clear(self):
        """Clear all stored messages and reset metadata counters."""
        self.messages.clear()
        self.metadata["total_queries"] = 0
        self.metadata["created_at"] = datetime.utcnow().isoformat()


class ConversationManager:
    """Manages multiple conversation sessions"""

    def __init__(self):
        self.conversations: Dict[str, ConversationHistory] = {}

    def create_conversation(
        self, conversation_id: str, system_prompt: str, max_messages: int = 20
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
            {"id": conv_id, "summary": conv.get_summary()}
            for conv_id, conv in self.conversations.items()
        ]


# =============================================================================
# UPDATED AGENT FUNCTION WITH HISTORY
# =============================================================================


async def process_query_stream_with_history(
    user_query: str,
    conversation: ConversationHistory,
    openai_client=None,
    execute_tool=None,
    TOOLS=None,
    Config=None,
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
        logger.info(
            f"[History] After adding user, messages={len(conversation.messages)}"
        )

        # Get conversation history for API (includes all previous context)
        messages = conversation.get_messages_for_api()

        logger.info(
            f"Sending {len(messages)} messages to API (including system prompt)"
        )
        for i, msg in enumerate(messages):
            logger.info(
                f"[History] msg[{i}] role={msg['role']} content={str(msg.get('content'))[:150]}"
            )
        #         # Add vector context if provided (only to current query, not history)
        #         if collection_name and vector_service:
        #             logger.info(f"Adding vector context from: {collection_name}")
        #             vector_result = vector_service.query(collection_name, user_query, limit=3)

        #             if vector_result['success'] and vector_result['data']:
        #                 context_docs = [doc['content'] for doc in vector_result['data']]
        #                 context = '\n\n'.join(context_docs)

        #                 # Add context as a separate system message (temporary, not saved to history)
        #                 context_message = {
        #                     "role": "system",
        #                     "content": f"""<context>
        # {context}
        # </context>

        # Use the context above if relevant to answer the current user query."""
        #                 }
        #                 # Insert before the last user message
        #                 messages.insert(-1, context_message)

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
                stream=True,
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
                                tool_calls.append(
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                )

                            current_tool_call = tool_calls[tc_delta.index]

                            if tc_delta.id:
                                current_tool_call["id"] = tc_delta.id

                            if tc_delta.function:
                                if tc_delta.function.name:
                                    current_tool_call["function"]["name"] = (
                                        tc_delta.function.name
                                    )
                                if tc_delta.function.arguments:
                                    current_tool_call["function"]["arguments"] += (
                                        tc_delta.function.arguments
                                    )

                # Handle content
                if delta.content:
                    current_content += delta.content

                    # Emit content character by character
                    for char in delta.content:
                        content_data = {"type": "content", "data": char}
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
                        "type": "tool_call",
                        "tool": tool_call["function"]["name"],
                        "args": json.loads(tool_call["function"]["arguments"]),
                        "id": tool_call["id"],
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
                        "content": tool_result,
                    }
                    messages.append(tool_msg)

                    # Emit tool result (truncated for display)
                    truncated_result = (
                        tool_result[:500] + "..."
                        if len(tool_result) > 500
                        else tool_result
                    )
                    tool_result_data = {
                        "type": "tool_result",
                        "tool": function_name,
                        "content": truncated_result,
                        "tool_call_id": tool_call["id"],
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
            full_assistant_content
        )

        logger.info(f"Conversation now has {len(conversation.messages)} messages")

        # Send completion
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        logger.info("Query processing complete")

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        error_data = {
            "type": "error",
            "message": str(e),
            "error_type": type(e).__name__,
        }
        yield f"data: {json.dumps(error_data)}\n\n"
