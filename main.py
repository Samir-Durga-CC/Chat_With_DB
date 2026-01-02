"""
PostgreSQL Chat Assistant
A simple chat interface to interact with PostgreSQL databases using OpenAI API
"""

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration - Update these with your credentials
DB_CONFIG = {
    'host': 'localhost',
    'database': 'alpha-product-samir',
    'user': 'postgres',
    'password': 'password',
    'port': 5432
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


class DatabaseService:
    """Handle PostgreSQL database operations"""
    
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            return True
        except Exception as e:
            print(f"{COLORS['red']}Failed to connect to database: {e}{COLORS['reset']}")
            return False
    
    def test_connection(self):
        """Test database connection"""
        try:
            self.cursor.execute('SELECT 1')
            return True
        except Exception as e:
            raise Exception(f"Connection test failed: {e}")
    
    def execute_query(self, query):
        """Execute SQL query and return results"""
        try:
            self.cursor.execute(query)
            
            # Check if query returns results
            if self.cursor.description:
                rows = self.cursor.fetchall()
                columns = [desc[0] for desc in self.cursor.description]
                return {
                    'rows': [dict(row) for row in rows],
                    'columns': columns,
                    'rowcount': self.cursor.rowcount,
                    'command': query.strip().split()[0].upper()
                }
            else:
                self.conn.commit()
                return {
                    'rows': [],
                    'columns': [],
                    'rowcount': self.cursor.rowcount,
                    'command': query.strip().split()[0].upper()
                }
        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Query execution failed: {e}")
    
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


def get_system_prompt(schema):
    """Generate system prompt with database schema"""
    return f"""
<role>
You are a helpful SQL assistant. You help users interact with PostgreSQL databases, 
generate queries based on requests, and retrieve data and information.
</role>

<rules>
    <rule>Always ask for user confirmation before executing Create, Update, Alter, or Delete queries.</rule>
    <rule>Try to use queries most of the time and execute them.</rule>
    <rule>When you suggest a query, always ask if the user wants you to execute it.</rule>
    <rule>PostgreSQL identifiers are case-sensitive. Any table or column name with uppercase letters MUST be wrapped in double quotes.</rule>
    <rule>Use the database schema EXACTLY as provided. Do not remove double quotes from identifiers.</rule>
    <rule>Always suggest best practices.</rule>
    <rule>Default limit on SELECT is 10 rows unless user asks for more.</rule>
    <rule>When listing data, use Markdown table format with headers and rows.</rule>
    <rule>Optimize queries for performance and readability.</rule>
</rules>

<general_guidelines>
    <rule>Don't answer questions unrelated to your role as an SQL Assistant.</rule>
    <rule>Be concise and helpful. Be brief, not verbose.</rule>
</general_guidelines>

<database>
Consider the following PostgreSQL database schema:
{schema}
</database>
"""


def format_table(rows, columns):
    """Format query results as Markdown table"""
    if not rows:
        return "No results found."
    
    # Create header
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(['---' for _ in columns]) + " |"
    
    # Create rows
    table_rows = []
    for row in rows:
        row_values = [str(row.get(col, '')) for col in columns]
        table_rows.append("| " + " | ".join(row_values) + " |")
    
    return "\n".join([header, separator] + table_rows)


def main():
    """Main chat loop"""
    print(f"{COLORS['green']}Welcome to PostgreSQL Chat Assistant!{COLORS['reset']}\n")
    
    # Initialize database service
    db = DatabaseService(DB_CONFIG)
    
    # Connect to database
    if not db.connect():
        return
    
    try:
        db.test_connection()
        print(f"{COLORS['green']}✓ Database connection successful!{COLORS['reset']}")
        print(f"Connected to: {DB_CONFIG['database']}@{DB_CONFIG['host']}\n")
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
    initial_message = "Give me an overview of the database, list the tables and their count. Tell me briefly how you can help me with this database."
    
    chat_history.append({'role': 'user', 'content': initial_message})
    
    # Database tools definition
    tools = [{
        'type': 'function',
        'function': {
            'name': 'execute_sql_query',
            'description': 'Execute a SQL query on the database. Use for SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER commands.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'sql_query': {
                        'type': 'string',
                        'description': 'The SQL query to execute'
                    },
                    'description': {
                        'type': 'string',
                        'description': 'Brief description of what the query does'
                    }
                },
                'required': ['sql_query', 'description']
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
    print(f"\n{COLORS['yellow']}Type 'exit' to quit, 'history' to see chat history{COLORS['reset']}")
    print("=" * 60)
    
    while True:
        try:
            user_input = input(f"\n{COLORS['blue']}You: {COLORS['reset']}").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print(f"{COLORS['green']}Goodbye!{COLORS['reset']}")
                break
            
            if user_input.lower() == 'history':
                print(f"\n{COLORS['yellow']}=== Chat History ==={COLORS['reset']}")
                for msg in chat_history:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '[No content]')
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
                    sql_query = function_args['sql_query']
                    
                    print(f"\n{COLORS['blue']}Executing: {sql_query}{COLORS['reset']}")
                    
                    try:
                        result = db.execute_query(sql_query)
                        
                        # Format result
                        if result['rows']:
                            print(f"\n{format_table(result['rows'], result['columns'])}")
                            print(f"\n{COLORS['yellow']}({result['rowcount']} rows){COLORS['reset']}")
                        else:
                            print(f"{COLORS['green']}Query executed successfully. {result['rowcount']} rows affected.{COLORS['reset']}")
                        
                        # Update schema if DDL command
                        if result['command'] in ['CREATE', 'ALTER', 'DROP']:
                            schema = db.get_schema()
                            print(f"{COLORS['blue']}Database schema updated{COLORS['reset']}")
                        
                        # Add tool result to history
                        tool_result = {
                            'role': 'tool',
                            'tool_call_id': tool_call.id,
                            'content': json.dumps(result)
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