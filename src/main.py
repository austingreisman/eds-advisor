from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader, load_index_from_storage, ServiceContext
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llama_pack import download_llama_pack
from ollama_pack.llama_index.packs.ollama_query_engine.base import OllamaQueryEnginePack
from ollama_pack.llama_index.packs.ollama_query_engine.base import OllamaEmbedding
from llama_index.core.schema import Document

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

# Initialize LlamaPack
embed_model = OllamaEmbedding(
            model_name=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL
        )
llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

# Set global settings
Settings.embed_model = embed_model
Settings.chunk_size = 1024
Settings.llm = llm

# Get file full file paths from docs folder
docs_dir = Path(__file__).parent.parent / "docs"
documents = [str(file_path) for file_path in docs_dir.glob("*.pdf")]
# Function to create or load indices
def setup_indices(documents):
    index_set = {}
    
    for doc in documents:
        doc_name = Path(doc).stem
        storage_path = Path(__file__).parent / "storage" / doc_name
        
        try:
            # Try to load existing index
            storage_context = StorageContext.from_defaults(persist_dir=str(storage_path))
            cur_index = load_index_from_storage(storage_context)
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
        # Read the contents of the text file
        description_path = Path(__file__).parent.parent / "docs" / f"{doc_name} description.txt"
        try:
            with open(description_path, 'r') as f:
                description = f.read()
        except FileNotFoundError:
            description = ""
        
        index_set[doc] = [cur_index, description]
    
    return index_set

def systematic_query(query):
    """
    Systematically query through all available tools if initial query fails
    """
    # First, try a general query
    initial_response = agent.chat(query)
    
    # If response is inconclusive, systematically check each tool
    if initial_response.sources[0].content == 'Empty Response':
        for tool in individual_query_engine_tools:
            try:
                specific_response = tool.query_engine.query(query)
                if specific_response.response != 'Empty Response':
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
        query_engine=index_set[doc][0].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_{Path(doc).name.replace(' ', '_')}",  # Use full filename
            description=index_set[doc][1]
        ),
    )
    for doc in documents
]

# Setup Agent

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
    1. Answer the question using only information from the documents by using your tools.".
    2. Do NOT use markdown.
    3. Keep your answers concise and to the point."""

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