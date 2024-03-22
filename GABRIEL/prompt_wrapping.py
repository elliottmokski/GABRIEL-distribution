import functools
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Assuming your Jinja2 templates are in the 'templates' directory
from GABRIEL import foundational_functions
import os

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