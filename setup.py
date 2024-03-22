from setuptools import setup, find_packages

setup(
    name='GABRIEL-ratings', 
    version='0.1.0',  
    author='Elliott Mokski',  
    author_email='elliottpmokski@gmail.com',  
    description='THE GABRIEL library for numerical analysis of texts in the social sciences.', 
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/elliottmokski/GABRIEL-distribution',  
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        # List your project's dependencies here as strings, for example:
        # 'numpy>=1.18.1',
        # 'requests>=2.23.0',
    ],
    classifiers=[
        # Classifiers help users find your project. Full list: https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',  # Specify the Python versions you support here
        'License :: OSI Approved :: MIT License',  # Choose the license for your package
        'Operating System :: OS Independent',  # Specify compatible OS
    ],
    python_requires='>=3.6',  # Minimum version requirement of Python
)