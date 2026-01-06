"""
MongoDB Chat Assistant
A simple chat interface to interact with MongoDB databases using OpenAI API
"""

import os
import json
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from bson import ObjectId, json_util
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Configuration - Docker setup
MONGO_CONFIG = {
    'connection_string': 'mongodb://root:password@localhost:27020/',
    'database': 'alpha-kcc' 
}

# OpenAI Configuration
OPENAI_MODEL = 'gpt-4o-mini'

# ANSI color codes
COLORS = {
    'reset': '\033[0m',
    'green': '\033[32m',
    'blue': '\033[34m',
    'red': '\033[31m',
    'yellow': '\033[33m'
}


class MongoDBService:
    """Handle MongoDB database operations"""
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self.db = None
    
    def connect(self):
        """Establish database connection"""
        try:
            # Use connection string directly
            connection_string = self.config['connection_string']
            
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            self.db = self.client[self.config['database']]
            return True
        except Exception as e:
            print(f"{COLORS['red']}Failed to connect to MongoDB: {e}{COLORS['reset']}")
            return False
    
    def test_connection(self):
        """Test database connection"""
        try:
            self.client.admin.command('ping')
            return True
        except ConnectionFailure as e:
            raise Exception(f"Connection test failed: {e}")
    
    def execute_query(self, operation):
        """Execute MongoDB operation and return results"""
        try:
            operation_dict = json.loads(operation)
            collection_name = operation_dict.get('collection')
            action = operation_dict.get('action')
            params = operation_dict.get('params', {})
            
            if not collection_name or not action:
                raise ValueError("Operation must specify 'collection' and 'action'")
            
            collection = self.db[collection_name]
            
            # Execute based on action
            if action == 'find':
                query = params.get('query', {})
                projection = params.get('projection', None)
                limit = params.get('limit', 10)
                sort = params.get('sort', None)
                
                cursor = collection.find(query, projection).limit(limit)
                if sort:
                    cursor = cursor.sort(sort)
                
                results = list(cursor)
                return {
                    'results': results,
                    'count': len(results),
                    'action': 'find'
                }
            
            elif action == 'findOne':
                query = params.get('query', {})
                projection = params.get('projection', None)
                result = collection.find_one(query, projection)
                return {
                    'results': [result] if result else [],
                    'count': 1 if result else 0,
                    'action': 'findOne'
                }
            
            elif action == 'insertOne':
                document = params.get('document', {})
                result = collection.insert_one(document)
                return {
                    'inserted_id': str(result.inserted_id),
                    'acknowledged': result.acknowledged,
                    'action': 'insertOne'
                }
            
            elif action == 'insertMany':
                documents = params.get('documents', [])
                result = collection.insert_many(documents)
                return {
                    'inserted_ids': [str(id) for id in result.inserted_ids],
                    'acknowledged': result.acknowledged,
                    'count': len(result.inserted_ids),
                    'action': 'insertMany'
                }
            
            elif action == 'updateOne':
                query = params.get('query', {})
                update = params.get('update', {})
                result = collection.update_one(query, update)
                return {
                    'matched_count': result.matched_count,
                    'modified_count': result.modified_count,
                    'acknowledged': result.acknowledged,
                    'action': 'updateOne'
                }
            
            elif action == 'updateMany':
                query = params.get('query', {})
                update = params.get('update', {})
                result = collection.update_many(query, update)
                return {
                    'matched_count': result.matched_count,
                    'modified_count': result.modified_count,
                    'acknowledged': result.acknowledged,
                    'action': 'updateMany'
                }
            
            elif action == 'deleteOne':
                query = params.get('query', {})
                result = collection.delete_one(query)
                return {
                    'deleted_count': result.deleted_count,
                    'acknowledged': result.acknowledged,
                    'action': 'deleteOne'
                }
            
            elif action == 'deleteMany':
                query = params.get('query', {})
                result = collection.delete_many(query)
                return {
                    'deleted_count': result.deleted_count,
                    'acknowledged': result.acknowledged,
                    'action': 'deleteMany'
                }
            
            elif action == 'aggregate':
                pipeline = params.get('pipeline', [])
                results = list(collection.aggregate(pipeline))
                return {
                    'results': results,
                    'count': len(results),
                    'action': 'aggregate'
                }
            
            elif action == 'countDocuments':
                query = params.get('query', {})
                count = collection.count_documents(query)
                return {
                    'count': count,
                    'action': 'countDocuments'
                }
            
            else:
                raise ValueError(f"Unsupported action: {action}")
                
        except Exception as e:
            raise Exception(f"Query execution failed: {e}")
    
    def get_schema(self):
        """Get database schema (collections and sample documents)"""
        try:
            collections = self.db.list_collection_names()
            
            schema_output = '-- MongoDB Database Schema\n\n'
            schema_output += f"Database: {self.config['database']}\n"
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
                    # Use json_util.dumps directly - it handles ObjectId and other BSON types
                    schema_output += json_util.dumps(sample, indent=2)
                    schema_output += "\n"
                
                # Get indexes
                indexes = collection.index_information()
                if indexes:
                    schema_output += f"Indexes: {', '.join(indexes.keys())}\n"
                
                schema_output += "\n" + "-" * 60 + "\n\n"
            
            return schema_output
        except Exception as e:
            raise Exception(f"Schema export failed: {e}")
        
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()


def get_system_prompt(schema):
    """Generate system prompt with database schema"""
    return f"""
<role>
You are a helpful MongoDB assistant. You help users interact with MongoDB databases, 
generate queries based on requests, and retrieve data and information.
</role>

<rules>
    
  
    <rule>MongoDB query syntax uses JSON format. Be precise with MongoDB operators like $eq, $gt, $gte, $lt, $lte, $in, $and, $or, etc.</rule>
    <rule>Default limit on find() is 10 documents unless user asks for more.</rule>
    <rule>When listing data, present it in a clear, readable JSON format.</rule>

    <rule>Use aggregation pipelines for complex queries involving grouping, sorting, or transformations.</rule>
</rules>

<mongodb_operations>
Supported operations format:
{{
    "collection": "collection_name",
    "action": "find|findOne|insertOne|insertMany|updateOne|updateMany|deleteOne|deleteMany|aggregate|countDocuments",
    "params": {{
        // Action-specific parameters
    }}
}}

Examples:
- Find: {{"collection": "users", "action": "find", "params": {{"query": {{"age": {{"$gt": 25}}}}, "limit": 10}}}}
- Insert: {{"collection": "users", "action": "insertOne", "params": {{"document": {{"name": "John", "age": 30}}}}}}
- Update: {{"collection": "users", "action": "updateOne", "params": {{"query": {{"name": "John"}}, "update": {{"$set": {{"age": 31}}}}}}}}
- Aggregate: {{"collection": "orders", "action": "aggregate", "params": {{"pipeline": [{{"$group": {{"_id": "$status", "count": {{"$sum": 1}}}}}}]}}}}
</mongodb_operations>

<general_guidelines>
    <rule>Don't answer questions unrelated to your role as a MongoDB Assistant.</rule>
    <rule>Be concise and helpful. Be brief, not verbose.</rule>
    <rule>When showing documents, format JSON clearly with proper indentation.</rule>
</general_guidelines>

<database>
Current MongoDB database schema:
{schema}
</database>
"""


def format_json(data):
    """Format data as pretty JSON"""
    try:
        # Convert MongoDB objects to JSON-serializable format
        json_str = json_util.dumps(data, indent=2)
        return json.dumps(json.loads(json_str), indent=2)
    except:
        return str(data)
def invoke_mongodb_operation(db_service, operation, description):
    
   
    # Initialize database service
    db = MongoDBService(MONGO_CONFIG)
    
    # Connect to database
    if not db.connect():
        return
    
    try:
        db.test_connection()
        print(f"{COLORS['green']}✓ MongoDB connection successful!{COLORS['reset']}")
        print(f"Connected to: {MONGO_CONFIG['database']} via Docker MongoDB on port 27020\n")
    except Exception as e:
        print(f"{COLORS['red']}Connection test failed: {e}{COLORS['reset']}")
        return
    
    # Get database schema
    print("Loading database schema...")
    schema = db.get_schema()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Initialize chat history
    chat_history = []
    
    # Get initial database overview
  
    
 
    
    # Database tools definition
    tools = [{
        'type': 'function',
        'function': {
            'name': 'execute_mongodb_operation',
            'description': 'Execute a MongoDB operation. Use for find, insert, update, delete, aggregate, and count operations.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'operation': {
                        'type': 'string',
                        'description': 'JSON string containing the MongoDB operation with collection, action, and params'
                    },
                    'description': {
                        'type': 'string',
                        'description': 'Brief description of what the operation does'
                    }
                },
                'required': ['operation', 'description']
            }
        }
    }]
    
    # Get initial response
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {'role': 'system', 'content': get_system_prompt(schema)},
            *chat_history
        ],
        tools=tools,
        tool_choice='auto'
    )
    
    assistant_message = response.choices[0].message
    return assistant_message

def main():
    """Main chat loop"""
    print(f"{COLORS['green']}Welcome to MongoDB Chat Assistant!{COLORS['reset']}\n")
    
    # Initialize database service
    db = MongoDBService(MONGO_CONFIG)
    
    # Connect to database
    if not db.connect():
        return
    
    try:
        db.test_connection()
        print(f"{COLORS['green']}✓ MongoDB connection successful!{COLORS['reset']}")
        print(f"Connected to: {MONGO_CONFIG['database']} via Docker MongoDB on port 27020\n")
    except Exception as e:
        print(f"{COLORS['red']}Connection test failed: {e}{COLORS['reset']}")
        return
    
    # Get database schema
    print("Loading database schema...")
    schema = db.get_schema()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Initialize chat history
    chat_history = []
    
    # Get initial database overview
    print(f"{COLORS['blue']}Getting database overview...{COLORS['reset']}\n")
    initial_message = "Give me an overview of the database, list the collections and their document counts. Tell me briefly how you can help me with this database."
    
    chat_history.append({'role': 'user', 'content': initial_message})
    
    # Database tools definition
    tools = [{
        'type': 'function',
        'function': {
            'name': 'execute_mongodb_operation',
            'description': 'Execute a MongoDB operation. Use for find, insert, update, delete, aggregate, and count operations.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'operation': {
                        'type': 'string',
                        'description': 'JSON string containing the MongoDB operation with collection, action, and params'
                    },
                    'description': {
                        'type': 'string',
                        'description': 'Brief description of what the operation does'
                    }
                },
                'required': ['operation', 'description']
            }
        }
    }]
    
    # Get initial response
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {'role': 'system', 'content': get_system_prompt(schema)},
            *chat_history
        ],
        tools=tools,
        tool_choice='auto'
    )
    
    assistant_message = response.choices[0].message
    
    if assistant_message.content:
        print(f"{COLORS['green']}{assistant_message.content}{COLORS['reset']}\n")
    
    chat_history.append(assistant_message.model_dump(exclude_unset=True))
    
    # Main chat loop
    print(f"\n{COLORS['yellow']}Type 'exit' to quit, 'history' to see chat history, 'schema' to refresh schema{COLORS['reset']}")
    print("=" * 60)
    
    while True:
        try:
            user_input = input(f"\n{COLORS['blue']}You: {COLORS['reset']}").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print(f"{COLORS['green']}Goodbye!{COLORS['reset']}")
                break
            
            if user_input.lower() == 'schema':
                print(f"\n{COLORS['yellow']}Refreshing schema...{COLORS['reset']}")
                schema = db.get_schema()
                print(schema)
                continue
            
            if user_input.lower() == 'history':
                print(f"\n{COLORS['yellow']}=== Chat History ==={COLORS['reset']}")
                for msg in chat_history:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '[No content]')
                    if isinstance(content, str):
                        print(f"{role}: {content[:100]}...")
                print(f"{COLORS['yellow']}==================={COLORS['reset']}\n")
                continue
            
            # Add user message to history
            chat_history.append({'role': 'user', 'content': user_input})
            
            # Get AI response
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {'role': 'system', 'content': get_system_prompt(schema)},
                    *chat_history
                ],
                tools=tools,
                tool_choice='auto'
            )
            
            assistant_message = response.choices[0].message
            
            # Display assistant response
            if assistant_message.content:
                print(f"\n{COLORS['green']}Assistant: {assistant_message.content}{COLORS['reset']}")
            
            # Add to history
            chat_history.append(assistant_message.model_dump(exclude_unset=True))
            
            # Handle tool calls
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    function_args = json.loads(tool_call.function.arguments)
                    operation = function_args['operation']
                    description = function_args.get('description', '')
                    
                    print(f"\n{COLORS['blue']}Executing: {description}{COLORS['reset']}")
                    print(f"{COLORS['yellow']}Operation: {operation}{COLORS['reset']}")
                    
                    try:
                        result = db.execute_query(operation)
                        
                        # Format and display result
                        if 'results' in result and result['results']:
                            print(f"\n{COLORS['green']}Results:{COLORS['reset']}")
                            print(format_json(result['results']))
                            print(f"\n{COLORS['yellow']}({result.get('count', 0)} documents){COLORS['reset']}")
                        else:
                            print(f"\n{COLORS['green']}{format_json(result)}{COLORS['reset']}")
                        
                        # Update schema if collection structure might have changed
                        if result.get('action') in ['insertOne', 'insertMany']:
                            schema = db.get_schema()
                            print(f"{COLORS['blue']}Schema refreshed{COLORS['reset']}")
                        
                        # Add tool result to history
                        tool_result = {
                            'role': 'tool',
                            'tool_call_id': tool_call.id,
                            'content': json.dumps(result, default=str)
                        }
                        chat_history.append(tool_result)
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        print(f"{COLORS['red']}{error_msg}{COLORS['reset']}")
                        
                        tool_result = {
                            'role': 'tool',
                            'tool_call_id': tool_call.id,
                            'content': error_msg
                        }
                        chat_history.append(tool_result)
                
                # Get follow-up response after tool execution
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {'role': 'system', 'content': get_system_prompt(schema)},
                        *chat_history
                    ],
                    tools=tools
                )
                
                if response.choices[0].message.content:
                    print(f"\n{COLORS['green']}Assistant: {response.choices[0].message.content}{COLORS['reset']}")
        
        except KeyboardInterrupt:
            print(f"\n{COLORS['green']}Goodbye!{COLORS['reset']}")
            break
        except Exception as e:
            print(f"{COLORS['red']}Error: {e}{COLORS['reset']}")
    
    # Cleanup
    db.close()


if __name__ == '__main__':
    main()