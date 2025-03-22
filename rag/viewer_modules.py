from langchain_community.llms import Ollama

llm_model_name = "llama3"

def get_llm_response(prompt, temperature=0):

    # Generate a response using the model

    # # Pull the model from the langchain_community
    # try:
    #     # Provide the full path to the ollama executable
    #     subprocess.run([ollama_executable, "pull", llm_name], check=True)
    #     print(f"Model {llm_name} pulled successfully.")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error pulling the model {llm_name}: {e}")
    # except FileNotFoundError as e:
    #     print(f"Executable not found: {e}")

    # Initialize the Ollama model
    model = Ollama(model=llm_model_name)

    # Fix the random seed for reproducibility
    # random.seed(42)
    # np.random.seed(42)
    # torch.manual_seed(42)

    # Generate a response using the model
    response = model.generate(prompts=[prompt], temperature=temperature)

    return response

def get_llm_response_string(prompt, temperature=0):

    response = get_llm_response(prompt, temperature)

    return response.generations[0][0].text
