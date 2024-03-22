import functools
import openai
import queue
import threading
import time
from jinja2 import Environment, FileSystemLoader, select_autoescape
from GABRIEL import foundational_functions

import os

# Determine the path to the 'Prompts' folder dynamically
package_dir = os.path.dirname(os.path.abspath(foundational_functions.__file__))
templates_dir = os.path.join(package_dir, 'Prompts')
env = Environment(
    loader=FileSystemLoader(templates_dir),
    autoescape=select_autoescape()
)

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
                except openai._exceptions.RateLimitError as e:
                    print(f"Rate limit error on attempt {attempt + 1}/{max_retries}. Retrying after delay.")
                    time.sleep(delay)  # Sleep and retry after delay
                except openai._exceptions.OpenAIError as e:
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

def call_api(rendered_prompt, system_instruction, timeout, temperature, model,desired_response_format = 'text', seed = None, api_key = None, **kwargs):
    assistant = ChatAssistant(model=model, api_key= api_key)
    return assistant.generate_response(
        rendered_prompt,
        system_instruction=system_instruction,
        timeout=timeout,
        temperature=temperature,
        desired_response_format = desired_response_format,
        seed = seed,
        **kwargs
    )

class ChatAssistant:
    def __init__(self, model, api_key):
        self.model = model
        self.client = openai.OpenAI(api_key = api_key)

    @retry_on_api_error(max_retries=3)
    def generate_response(self, prompt, system_instruction, external_messages=None, timeout=100, 
                          temperature=0.9, max_tokens=1500, desired_response_format='text', seed = None):
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

            except openai._exceptions.RateLimitError:
                time.sleep(45)
                raise
            except openai._exceptions.OpenAIError as e:
                raise
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