# Run streamlit viewer using bootstrap

from streamlit import bootstrap

real_script = 'viewer_main.py'

bootstrap.run(real_script, f'viewer_main.py {real_script}', [], {})