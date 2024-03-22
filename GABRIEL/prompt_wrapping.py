import functools
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Assuming your Jinja2 templates are in the 'templates' directory
env = Environment(
    loader=FileSystemLoader('Gabriel_2/Prompts'),
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