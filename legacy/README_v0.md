# The Generalized Attribute Based Ratings Information Extraction Library (GABRIEL)

## Description

GABRIEL is a simple Python framework built on top of LLMs like ChatGPT to simplify quantitative text analysis in the social sciences.

IMPORTANT: Follow this Colab tutorial notebook for the easiest setup guide: https://colab.research.google.com/drive/1tshfY-2al7asU7pTFvFFg1n4NSvLXtZg?usp=sharing

The full documentation is below.

## Installation

The new Python library replaces the previous API and dramatically simplifies the use of the package. Installation is extremely simple using pip.

Before you install our package, we require that you install the OpenAI library. Open your terminal or command prompt and run:

```bash
pip install openai
```

Once you have installed OpenAI, install GABRIEL using 
```bash
pip install gabriel-ratings
``` 

## Use
### Simple ratings framework

The main way to get ratings from GABRIEL is using the Archangel class. The class requires an OpenAI api-key for instantiation. We strongly recommend you store the key as an environment variable. To create an Archangel object, use the following syntax. 

```python
from GABRIEL.Archangel import Archangel
combined_assistant = Archangel(your_api_key)
```

Once you create the object, you can run a simple ratings framework through the *rate_texts* function. You must supply a list of the texts to rate, *texts*; an *attributes_dict*, where the keys are your attributes, and the values are the definitions, and a *task_description*, which is a few sentence description of what you're trying to acccomplish (your data, your question, etc.). In addition, we require a *save_folder* and a *file_name*, which is where the output from your run will be saved.

You can also specify a specific OpenAI model for your call, using the *model* parameter (the default is GPT-3.5-turbo). See below for the full list of parameters, and more detailed descriptions.

The simplest ratings call, which returns a Pandas dataframe, is just:

```python
ratings = archangel.rate_texts(texts, attribute_dict= attribute_dict, save_folder = 'path_to_your_folder', file_name = 'your_file_name.csv', task_description = 'your_task_description')
```

### Features 

The Archangel class comes with a number of easy to use features to help you run your code. 
- parallelization: the library parallelizes API calls to dramatically speed up running time. We configure this by default.
- cost estimates: we provide a very rough cost estimate of each run when you begin the call, based on the model and texts you input. 
- auto-saving: the class will auto-save your results to a CSV at each iteration, as long as you provide a valid path.  

### Preset classes

To simplify the task of choosing your hyperparameters, we provide two default options: 
- 'mazda': cheap, fast, and reliable. Uses GPT-3.5-turbo, with text truncation to 9500 words to allow for prompts. Runs 50 queries in parallel.  
- 'tesla': expensive. Uses GPT-4-turbo, with 30 parallel queries. Not recommended due to cost. 

### Function parameters

The full list of parameters for the function is as follows. 

- **`attribute_dict`** A dictionary where the keys are the attributes you want to evaluate, and the values are the descriptions. See Colab notebook for examplse. 
<!-- - **`attributes`** (optional): A list containing the desired attributes for evaluation. If this is not specified, the model will generate **`n_search_attributes`** attributes itself. For example, `n_search_attributes = ['optimism', 'negativity', 'concern about unemployment']`. -->
<!-- - **`n_search_attributes`** (optional): An integer containing the number of attributes to generate if no attributes were specified. Defaults to 5. For example, `n_search_attributes = 10`. -->
<!-- - **`descriptions`** (optional): A list of descriptions for the attributes if **`attributes`** are provided explicitly. Otherwise, the descriptions will be generated by the model. For example, `descriptions = ['The happy attribute refers to a positive emotional state characterized by feelings of joy, contentment, and satisfaction.', 'The "sad" attribute is a feeling of sorrow, unhappiness, or distress. It is an emotional state characterized by a low mood and a sense of loss or disappointment.']`. -->
<!-- - **`object_clarification`** (optional): A string providing further clarification on the objects (e.g., a string of comma-separated examples). -->
<!-- - **`attribute_clarification`** (optional): A string providing further clarification on the attributes (e.g., a string of comma-separated examples). For use in the generation of attributes. -->
<!-- - **`use_classification`** (optional - defaults to False): Toggles whether the model uses a ratings or classification approach. -->
<!-- - **`classification_clarification`** (optional - only considered when `use_classification = True`): An additional string to provide context on the classification process. -->
- **`truncate`** (optional, defaults to True): Whether to truncate the text. This avoids overloading the API token limit (16k tokens for the default model).
-**`truncate_len`** (optional, defaults to 5000) the amount of text to keep. 
- **`project_probs`** (optional, defaults to False): Whether to project the probabilities from 0 to 100 to a 0 to 1 scale.
- **`api_key`** (mandatory): Your OpenAI API key.
- **`model`** (optional): Backend model, default = `o4-mini`.
- **`seed`** (optional, RECOMMENDED): Set a seed for cross-run replicability. For instance, `seed = 0`.

## Citation

Please cite the project using: 

The Generalized Attribute Based Ratings Information Extraction Library (GABRIEL). Hemanth Asirvatham and Elliott Mokski (2023). https://github.com/elliottmokski/GABRIEL-distribution. 
