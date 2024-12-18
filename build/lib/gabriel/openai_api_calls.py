import functools
import openai
import queue
import threading
import time
from jinja2 import Environment, FileSystemLoader, select_autoescape
from gabriel import foundational_functions
import json
import os
import platform
import pandas as pd

# Determine the path to the 'Prompts' folder dynamically
package_dir = os.path.dirname(os.path.abspath(foundational_functions.__file__))
templates_dir = os.path.join(package_dir, 'Prompts')

env = Environment(
    loader=FileSystemLoader(templates_dir),
    autoescape=select_autoescape()
)

def get_default_save_path():
    """Returns the default save path based on the operating system."""
    home_dir = os.path.expanduser("~")
    app_dir = "GABRIEL Batch Calls"
    
    if platform.system() == "Windows":
        return os.path.join(home_dir, "AppData", "Local", app_dir)
    elif platform.system() == "Darwin":  # macOS
        return os.path.join(home_dir, "Library", "Application Support", app_dir)
    else:  # Unix/Linux
        return os.path.join(home_dir, ".local", "share", app_dir)

class APIError(Exception):
    pass

def with_prompt(template_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            # The context for the template is now all arguments bound to their names
            template_context = kwargs
            
            # Print the context for debugging purposes
            # print("Template Context:", template_context)

            # Load and render the template
            template = env.get_template(f"{template_name}.j2")
            rendered_prompt = template.render(**template_context)
            
            # print("Rendered Prompt:", rendered_prompt)
            
            # Execute the wrapped function with the rendered prompt and any remaining kwargs
            return func(rendered_prompt=rendered_prompt, **kwargs)
        return wrapper
    return decorator

def retry_on_api_error(max_retries, delay=45):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except openai.RateLimitError as e:
                    print(f"Rate limit error on attempt {attempt + 1}/{max_retries}. Retrying after delay.")
                    time.sleep(delay)  # Sleep and retry after delay
                except openai.OpenAIError as e:
                    print(f"OpenAI error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt == max_retries - 1:
                        raise APIError("Failed after max retries due to OpenAI error.")
                except Exception as e:
                    print(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt == max_retries - 1:
                        raise APIError("Failed after max retries due to an unexpected error.")
            return ""
        return wrapper
    return decorator

def call_api(rendered_prompt, system_instruction, timeout, temperature, model,desired_response_format = 'text', seed = None, api_key = None, use_batch = False, **kwargs):
    assistant = ChatAssistant(model=model, api_key= api_key)
    
    if use_batch:
        # print('Batching')
        response = assistant.generate_batch_call(
            prompt=rendered_prompt,
            system_instruction=system_instruction,
            external_messages=kwargs.get('external_messages'),
            seed=seed,
            desired_response_format=desired_response_format,
            temperature=temperature,
            max_tokens=kwargs.get('max_tokens', 2000),
            custom_id=kwargs.get('custom_id', 'request-1'),
            file_name=kwargs.get('file_name', 'batch_requests')
        )
    else:
        response = assistant.generate_response(
            prompt=rendered_prompt,
            system_instruction=system_instruction,
            external_messages=kwargs.get('external_messages'),
            timeout=timeout,
            temperature=temperature,
            max_tokens=kwargs.get('max_tokens', 2000),
            desired_response_format=desired_response_format,
            seed=seed
        )
    return response

class ChatAssistant:
    def __init__(self, model, api_key):
        self.model = model
        self.client = openai.OpenAI(api_key = api_key)

    @retry_on_api_error(max_retries=3)
    def generate_response(self, prompt, system_instruction, external_messages=None, timeout=100, 
                          temperature=0.9, max_tokens=2000, desired_response_format='text', seed = None):
        response_queue = queue.Queue()

        def target():
            nonlocal response_queue, external_messages

            # Compose messages for the API call
            messages = [{"role": "system", "content": system_instruction}]
            if external_messages:
                messages.extend(external_messages)
            if prompt:
                messages.append({"role": "user", "content": prompt})

            # Convert the desired_response_format string to the correct dictionary format
            response_format = {"type": desired_response_format}

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                    seed = seed
                )
                # Extract the response content based on the specified format
                # if desired_response_format == 'text':
                response_message = response.choices[0].message.content
                # else: # If the response format is 'json', handle accordingly
                    # response_message = response.choices[0].message

            except openai.RateLimitError as e:
                raise e
            except openai.OpenAIError as e:
                raise e
            response_queue.put(response_message)

        # Start the thread to make the API call
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)

        # Retrieve the response from the queue
        if not response_queue.empty():
            return response_queue.get()
        else:
            raise APIError("API call timed out.")
        
    def generate_batch_call(self, prompt, system_instruction, external_messages = None, seed = None, desired_response_format='text', temperature=0.9, max_tokens=2000, custom_id = 'request-1', file_name = 'batch_requests'):
        batch_prompt = {"custom_id":custom_id,"method":"POST","url":"/v1/chat/completions"}
        messages = [{"role": "system", "content": system_instruction}]
        messages.append({"role": "user", "content": prompt})
        response_format = {"type": desired_response_format}

        batch_prompt["body"] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "response_format": response_format,
        }

        save_path = get_default_save_path()
        # batch_calls_dir = os.path.join(save_path, 'Batch Calls')
        # Check if 'Batch Calls' directory exists, if not create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_path = os.path.join(save_path, file_name) + '.jsonl'
        # print(f"Saving batch request to {file_path}")

        json_line = json.dumps(batch_prompt)
        with open(file_path, 'a') as f:
            f.write(json_line + "\n") 

        return file_path

class BatchRunner:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key = api_key)

    def run_batch(self, batch_name, description):
        file_path = os.path.join(get_default_save_path(), f"{batch_name}.jsonl")
        batch_input_file = self.client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        
        val = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": description
            }
    )
        
        return val
    
    def retrieve_batch(self, file_name, format = 'json'):
        batch_save_path = file_name.split('.')[0] + '_batch_metadata.csv'
        batch_metadata = pd.read_csv(batch_save_path, index_col = 0)
        batch_id = batch_metadata.loc['id','Value']
        batch_content = self.client.batches.retrieve(batch_id)
        print('The status of your batch is:', batch_content.status)
        if batch_content.status == 'completed':
            id = batch_content.output_file_id
            result = self.client.files.content(id).content
            save_path = get_default_save_path()
            result_file_name = os.path.join(save_path, file_name) + '_output.jsonl'
            with open(result_file_name, 'wb') as file:
                file.write(result)

            results = []
            with open(result_file_name, 'r') as file:
                for line in file:
                    # Parsing the JSON string into a dict and appending to the list of results
                    json_object = json.loads(line.strip())
                    results.append(json_object)
            final_df = pd.DataFrame()
            for res in results:
                task_id = res['custom_id']
                # Getting index from task id
                # index = task_id.split('-')[-1]
                try: 
                    result = res['response']['body']['choices'][0]['message']['content']
                    if format == 'json':
                        ratings = json.loads(result)
                        output_df = pd.DataFrame.from_dict(ratings, orient = 'index').T
                        # print(ratings)
                        # print(output_df)
                        output_df['custom_id'] = task_id
                        # output_df = output_df.set_index('Text').reset_index()
                        final_df = pd.concat([final_df, output_df], axis=0)
                # print(result)
                except:
                    pass

            raw_df = pd.read_csv(file_name)[['Text','custom_id']]
            raw_df['custom_id'] = raw_df['custom_id'].astype(str)
            final_df['custom_id'] = final_df['custom_id'].astype(str)
            final_df = final_df.merge(raw_df, on = 'custom_id')
            final_df.drop(columns = ['custom_id']).to_csv(file_name, index = False)
            return final_df.drop(columns = ['custom_id'])
