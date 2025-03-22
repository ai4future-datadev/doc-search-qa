import streamlit as st

import viewer_modules

st.title('Hello World!')

prompt = st.text_input('Enter a prompt:', 'Translate this text to Spanish: Hello, how are you?')

st.write('Generating response...')
response_string = viewer_modules.get_llm_response_string(prompt)
print('Responded.')

st.write(response_string)
