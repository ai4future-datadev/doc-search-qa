# Run streamlit viewer using bootstrap

from streamlit import bootstrap

real_script = 'streamlit_ollama_app.py'

bootstrap.run(real_script, f'streamlit_ollama_app.py {real_script}', [], {})