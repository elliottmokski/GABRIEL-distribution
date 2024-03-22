import functools
import openai
from jinja2 import Environment, FileSystemLoader, select_autoescape
import time 
import os

class APIError(Exception):
    pass

from GABRIEL import foundational_functions

# Determine the path to the 'Prompts' folder dynamically
package_dir = os.path.dirname(os.path.abspath(foundational_functions.__file__))
templates_dir = os.path.join(package_dir, 'Prompts')

env = Environment(
    loader=FileSystemLoader(templates_dir),
    autoescape=select_autoescape()
)

def with_prompt(template_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract additional context for the template from kwargs
            context = kwargs.pop('template_context', {})
            
            # Load and render the template
            template = env.get_template(f"{template_name}.j2")
            prompt = template.render(**context)
            
            # Execute the wrapped function with the rendered prompt and any remaining kwargs
            return func(prompt, *args, **kwargs)
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

