from setuptools import setup, find_packages

setup(
    name='GABRIEL-ratings', 
    version='0.1.0',  
    author='Hemanth Asirvatham and Elliott Mokski',  
    author_email='elliottpmokski@gmail.com',  
    description='THE GABRIEL library for numerical analysis of texts in the social sciences.', 
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/elliottmokski/GABRIEL-distribution',  
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'GABRIEL': ['Prompts/*.j2'],
    },
    install_requires=[
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        'openai>=1.0.0',
        'Jinja2>=2.11.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'],
    python_requires='>=3.6'
)