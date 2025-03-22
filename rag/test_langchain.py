from langchain_community.llms import Ollama

model_llm_name = "llama3"

model_llm = Ollama(model=model_llm_name)

def query_llm(prompt):
    response = model_llm.generate(prompts=[prompt])
    return response

# Test the query_llm function
response = query_llm("Translate this text to Spanish: Hello, how are you?")
print(response)
