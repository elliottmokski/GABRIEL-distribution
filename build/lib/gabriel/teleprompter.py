import os
from jinja2 import Environment, FileSystemLoader, select_autoescape
from gabriel import foundational_functions  # Ensure foundational_functions.py is in the same package

class Teleprompter:
    def __init__(self):
        # Dynamically determine the path to the 'Prompts' folder
        package_dir = os.path.dirname(os.path.abspath(foundational_functions.__file__))
        templates_dir = os.path.join(package_dir, 'Prompts')

        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape()
        )

    def clean_json_prompt(self, dirty_json_output, format_template_name):
        format_template = self.env.get_template(format_template_name).render()
        template = self.env.get_template('clean_json_prompt.j2')
        prompt = template.render(dirty_json_output=dirty_json_output, format_template=format_template)
        return prompt

    def attribute_description_prompt(self, attribute, attribute_category, description_length):
        template = self.env.get_template('attribute_description_prompt.j2')
        prompt = template.render(attribute=attribute, attribute_category=attribute_category, description_length=description_length)
        return prompt

    def list_generation_prompt(self, category, n_items, mode='item', 
                               object_clarification=None, attribute_clarification=None):
        template = self.env.get_template('list_generation_prompt.j2')
        prompt = template.render(category=category, n_items=n_items, mode=mode, 
                                 object_clarification=object_clarification, attribute_clarification=attribute_clarification)
        return prompt

    def ratings_prompt(self, attributes, descriptions, passage, object_category, attribute_category, 
                       classification_clarification=None, format='json'):
        template = self.env.get_template('ratings_prompt.j2')
        prompt = template.render(attributes=attributes, descriptions=descriptions, passage=passage,
                                 object_category=object_category, attribute_category=attribute_category,
                                 classification_clarification=classification_clarification, format=format)
        return prompt

    def ratings_prompt_full(self, attribute_dict, passage, entity_category, attribute_category, attributes, format='json'):
        template = self.env.get_template('ratings_prompt_full.j2')
        prompt = template.render(attribute_dict=attribute_dict, passage=passage, entity_category=entity_category, 
                                 attribute_category=attribute_category, attributes=attributes, format=format)
        return prompt

    def classification_prompt(self, attributes, descriptions, passage, object_category, attribute_category, 
                              classification_clarification=None, format='json'):
        template = self.env.get_template('classification_prompt.j2')
        prompt = template.render(attributes=attributes, descriptions=descriptions, passage=passage,
                                 object_category=object_category, attribute_category=attribute_category,
                                 classification_clarification=classification_clarification, format=format)
        return prompt

    def identify_categories_prompt(self, task_description, format='json'):
        template = self.env.get_template('identify_categories_prompt.j2')
        prompt = template.render(task_description=task_description, format=format)
        return prompt

    def generic_classification_prompt(self, entity_list, possible_classes, class_definitions, entity_category, output_format='json'):
        format_template = self.env.get_template("type_of_tech_format.j2").render()
        template = self.env.get_template('generic_classification_prompt.j2')
        prompt = template.render(
            entity_list=entity_list,
            possible_classes=possible_classes,
            class_definitions=class_definitions,
            entity_category=entity_category,
            output_format=format_template
        )
        return prompt

teleprompter = Teleprompter()