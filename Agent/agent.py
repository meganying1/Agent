import llama_cpp
from transformers import ReactCodeAgent
from agent_helper import SEARCH_PROMPT, WikipediaSearch, ArxivSearch, SearchByProperty, SearchByMaterial

##########################################################################################

def create_agent(repo_id, filename):
    # Create LLM object
    llm = llama_cpp.Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        n_ctx=4096,
        gpu=True,
        metal=True,
        n_gpu_layers=-1
    )

    # Define a function to call the LLM and return output
    def llm_engine(messages, max_tokens=1000, stop_sequences=['Task']) -> str:
        response = llm.create_chat_completion(
            messages=messages,
            stop=stop_sequences,
            max_tokens=max_tokens,
            temperature=0.6
        )
        answer = response['choices'][0]['message']['content']
        return answer
    
    # Create agent equipped with search tools
    websurfer_agent = ReactCodeAgent(
    system_prompt=SEARCH_PROMPT,
    tools=[WikipediaSearch(), ArxivSearch(), SearchByProperty(), SearchByMaterial()],
    llm_engine=llm_engine,
    add_base_tools = False,
    verbose = True,
    max_iterations=10
    )

    return websurfer_agent