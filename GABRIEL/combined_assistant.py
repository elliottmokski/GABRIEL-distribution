from GABRIEL.foundational_functions import *
import json
import pandas as pd
import numpy as np

class CombinedAssistant:
    def __init__(self, api_key, drive_folder = None, model='gpt-3.5-turbo-0125'):
        self.drive_folder = drive_folder
        self.api_key = api_key
        self.model = model

    def get_attribute_descriptions(self, attributes, attribute_category, description_length, timeout=75, temperature=0.8, model='gpt-3.5-turbo-0125', seed = None, api_key = None):
        if api_key is None:
            api_key = self.api_key
        descriptions = list()
        for attribute in attributes:
            curr = get_description_for_attribute(attribute=attribute, attribute_category=attribute_category, description_length=description_length, temperature = temperature, 
                                                 model = model, timeout = timeout, seed = seed, api_key = api_key)
            print(curr)
            descriptions.append(curr)
        return descriptions

    def simple_evaluation_pipeline(self, search_axis_1, object_category=None, attribute_category=None, descriptions=None,
                                object_clarification='', attribute_clarification='', attributes=None, 
                                description_length=75, n_search_attributes = 5, temperature=0.8,
                                use_classification = False, format = 'json', classification_clarification = None, 
                                project_probs = False, truncate = True,
                                model = 'gpt-3.5-turbo-0125', seed = None, api_key = None, truncate_len = 9500, timeout = 75):
        if api_key is None:
            api_key = self.api_key
        if attributes is None or attributes == False:
            attributes = generate_category_items(category = attribute_category, n_items=n_search_attributes, 
                                                 attribute_clarification=attribute_clarification, mode='attribute', temperature = temperature, seed = seed, api_key = api_key)
        attributes =  [item.lower() for item in attributes]
        print('Attributes extracted')

        print('Attributes',attributes)

        if descriptions is None:
            # descriptions = self.get_attribute_descriptions(attributes = attributes, attribute_category = attribute_category, temperature=temperature, description_length=description_length, seed = seed, api_key = api_key)
            descriptions = list()
            for attribute in attributes:
                curr = get_description_for_attribute(attribute=attribute, attribute_category=attribute_category, description_length=description_length, temperature = temperature, 
                                                    model = model, timeout = timeout, seed = seed, api_key = api_key)
                descriptions.append(curr)
        print(descriptions)
        print('Descriptions extracted, running ratings.')

        dfs = list()

        if use_classification:
            rating_function = generate_simple_classification
            attribute_param = 'classification category'
        else:
            rating_function = generate_simple_ratings
            attribute_param = 'attribute'

        for passage in search_axis_1:
            if truncate:
                try:
                    passage = ' '.join(passage.split()[:truncate_len])
                except:
                    pass

            try:
                raw_ratings = rating_function(attributes = attributes, descriptions = descriptions,
                                                    passage = passage, object_category= object_category, 
                                                    attribute_category=attribute_category,format = format,classification_clarification = classification_clarification,
                                                    temperature = temperature, model = model, seed = seed, api_key = api_key)
                
                if format == 'json':
                    ratings = json.loads(raw_ratings)['data']
                    # Create a dictionary for the current passage's ratings
                    passage_data = {'Text': passage}
                    for rating in ratings:
                        attribute = rating[attribute_param]
                        rating_value = rating['rating']
                        if project_probs:
                            rating_value = float(rating_value) / 100
                            rating_value = 1 / (1 + np.exp(-24 * (rating_value - 0.5)))
                        passage_data[attribute] = rating_value

                    dfs.append(passage_data)
            except:
                pass
        output_df = pd.DataFrame(dfs)
        print(output_df)
        return output_df