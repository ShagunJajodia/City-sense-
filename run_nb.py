import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
with open('CitySense_Complete_Guide.ipynb', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': '.'}})
with open('CitySense_Complete_Guide.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('Notebook executed successfully.')
