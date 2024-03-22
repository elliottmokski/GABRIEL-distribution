# from decorators import with_prompt
from GABRIEL.openai_api_calls import *
import re

@with_prompt('attribute_description_prompt')
def get_description_for_attribute(rendered_prompt, attribute, attribute_category, description_length, 
                                  timeout=75, temperature=0.8, model='gpt-3.5-turbo-0125', seed = None, api_key = None, **kwargs):
    system_instruction = "Please provide a brief description."

    # Call the API with the rendered prompt and other parameters
    description = call_api(
        rendered_prompt,
        system_instruction=system_instruction,
        timeout=timeout,
        temperature=temperature,
        model=model,
        seed = seed,
        api_key = api_key,
        **kwargs
    )

    return description.strip()

@with_prompt('list_generation_prompt')
def generate_category_items(rendered_prompt, category, n_items, mode='item', 
                            object_clarification=None, attribute_clarification=None, 
                            timeout=90, temperature=0.8, model='gpt-3.5-turbo-0125', 
                            seed = None, api_key = None, **kwargs):
    system_instruction = "Please list the items or attributes."

    # Call the API with the rendered prompt and other parameters
    response = call_api(
        rendered_prompt,
        system_instruction=system_instruction,
        timeout=timeout,
        temperature=temperature,
        model=model,
        seed = seed,
        api_key = api_key,
        **kwargs
    )

    print("API Response:", response)
    
    # Process the response to extract items
    items = re.split(r',\s|\n', response)
    items = [re.sub(r'\d+\.\s', '', item).strip() for item in items]  # Remove numbering and strip whitespace
    items = [re.sub(r'[^a-zA-Z0-9\s]', '', item) for item in items]  # Remove punctuation and special characters

    return items

@with_prompt('ratings_prompt')
def generate_simple_ratings(rendered_prompt, attributes, descriptions, passage,
                            object_category, attribute_category,format = 'json', classification_clarification = None,
                            timeout=90, temperature=0.8, model='gpt-3.5-turbo-0125',
                            seed = None, api_key = None, **kwargs):
    if format == 'json':
        desired_response_format = 'json_object'
        system_instruction = 'Please output precise ratings as requested, following the detailed JSON template.'
    else:
        desired_response_format = 'text'
        system_instruction = 'Please output precise ratings as requested, using the provided format.'

    response = call_api(rendered_prompt,
                        system_instruction,
                        timeout = timeout, 
                        temperature = temperature, 
                        model = model,
                        desired_response_format = desired_response_format,
                        seed = seed,
                        api_key = api_key,
                        **kwargs)
    
    return response

@with_prompt('classification_prompt')
def generate_simple_classification(rendered_prompt, attributes, descriptions, passage,
                            object_category, attribute_category,format = 'json',classification_clarification = None,
                            timeout=90, temperature=0.8, model='gpt-3.5-turbo-0125',
                            seed = None, api_key = None, **kwargs):
    if format == 'json':
        desired_response_format = 'json_object'
        system_instruction = 'Please output precise ratings as requested, following the detailed JSON template.'
    else:
        desired_response_format = 'text'
        system_instruction = 'Please output precise ratings as requested, using the provided format.'

    response = call_api(rendered_prompt,
                        system_instruction,
                        timeout = timeout, 
                        temperature = temperature, 
                        model = model,
                        desired_response_format = desired_response_format,
                        seed = seed,
                        api_key = api_key,
                        **kwargs)
    
    return response