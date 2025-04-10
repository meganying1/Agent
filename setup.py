from setuptools import setup, find_packages

setup(
    name='agent',
    version='0.1.0',
    author='Megan Ying',
    author_email='mying@andrew.cmu.edu',
    description='An agent equipped with tools for searching Wikipedia, Arxiv, and a material property database.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/meganying1/agent',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'llama-cpp-python',
        'transformers[agents]',
        'torch',
        'langchain',
        'langchain-community',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    package_data={
        'agent': ['material_properties_minmax.csv']
    },
)