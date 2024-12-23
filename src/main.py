from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import SentenceSplitter

# Set chunk size for indexing
Settings.chunk_size = 1024

# Get file full file paths from docs folder
docs_dir = Path(__file__).parent.parent / "docs"
documents = [str(file_path) for file_path in docs_dir.glob("*")]

# Function to create or load indices
def setup_indices(documents):
    index_set = {}
    
    for doc in documents:
        doc_name = Path(doc).stem
        storage_path = Path(__file__).parent / "storage" / doc_name
        
        try:
            # Try to load existing index
            storage_context = StorageContext.from_defaults(persist_dir=str(storage_path))
            cur_index = VectorStoreIndex.from_documents(
                [],  # Empty list since we're loading from storage
                storage_context=storage_context
            )
            print(f"Loaded existing index for {doc_name}")
        except:
            # If no existing index, create a new one
            print(f"Creating new index for {doc_name}")
            
            # Load documents using SimpleDirectoryReader
            documents = SimpleDirectoryReader(input_files=[doc]).load_data()
            
            # Use sentence splitter from the original function
            splitter = SentenceSplitter(chunk_size=1024)
            nodes = splitter.get_nodes_from_documents(documents)
            
            # Create storage context and index
            storage_context = StorageContext.from_defaults()
            cur_index = VectorStoreIndex(nodes, storage_context=storage_context)
            
            # Persist the new index
            storage_context.persist(persist_dir=str(storage_path))
        
        index_set[doc] = cur_index
    
    return index_set

def systematic_query(query):
    """
    Systematically query through all available tools if initial query fails
    """
    # First, try a general query
    initial_response = agent.chat(query)
    
    # If response is inconclusive, systematically check each tool
    if "I'm sorry" in str(initial_response) or len(str(initial_response)) < 50:
        for tool in individual_query_engine_tools:
            try:
                specific_response = tool.query_engine.query(query)
                if len(str(specific_response)) > 50:
                    return specific_response
            except Exception as e:
                print(f"Error querying tool {tool.metadata.name}: {e}")
        
        # If no tool provides a good answer
        return "I'm sorry, I couldn't find a comprehensive answer in my documents."
    
    return initial_response
# Setup indices
index_set = setup_indices(documents)

# Prepare query engine tools
individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index_set[doc].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_{Path(doc).stem}",
            description=f"useful for when you want to answer queries about {Path(doc).stem}",
        ),
    )
    for doc in documents
]

# Setup LLM and Agent
llm = Ollama(model="llama3.2", base_url="http://localhost:11434")

agent_worker = FunctionCallingAgentWorker.from_tools(
    individual_query_engine_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

# Chatbot Loop
def chat_loop():
    print("Hypermobile EDS and POTS Expert Chatbot")
    print("Type 'exit' to end the conversation")
    
    # Initial system prompt
    system_message = """Your name is Alice, you are a hypermobile ehlers-danlos syndrome (hEDS) and postural orthostatic tachycardia syndrome (POTs)
    expert. You understand that hEDS and POTs are difficult diseases to manage, and you are considerate to the user asking questions.
    You have access a database of documents on ehlers-danlos syndrome and postural orthostatic tachycardia syndrome.
    You MUST do the following:
    1. Answer the question using only information from the documents. If you cannot find the answer in the documents, say "I'm sorry, I don't have that information in my databases".
    2. Use the tools to answer the question.
    3. Mention that you are not a doctor and any advice given is for educational purposes only.
    4. Output text in normal English, not markdown.
    5. Keep your answers concise and to the point."""

    agent.chat_history.append({
        "role": "system", 
        "content": system_message
    })
    while True:
        user_input = input("User: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Query the agent
        response = systematic_query(user_input)
        print(str(response))

# Run the chat loop
if __name__ == "__main__":
    chat_loop()