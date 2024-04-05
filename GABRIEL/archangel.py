from GABRIEL.foundational_functions import *
# from GABRIEL.combined_assistant import CombinedAssistant
from combined_assistant import CombinedAssistant
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import json

class Archangel():
    def __init__(self, api_key) -> None:
        self.attributor = CombinedAssistant(api_key)
        self.api_key = api_key

    def estimate_cost(self, texts, model):
        if 'gpt-3.5' in model:
            return 2 * round(sum([len(text.split()) for text in texts]) / 1000 / 1000 * 1.33 * 0.5, 3)
        elif 'gpt-4' in model:
            return 2* round(sum([len(text.split()) for text in texts]) / 1000 / 1000 * 1.33 * 20,3)
        else:
            return 'Unknown model'

    def get_preset_params(self, preset='mazda'):
        if preset == 'mazda':
            return {'model': 'gpt-3.5-turbo', 'n_parallel': 50, 'num_runs': 10, 'rate_first': False, 'temperature':0.8,
                    'timeout': 75, 'truncate_len': 9500, 'seed': None, 'truncate':True, 'format':'json'}

    def get_attributes(self, attributes_dict=None, attribute_mode='compare'):
        attribute_dict = self.attributor.get_attributes(attributes_dict=attributes_dict, attribute_mode=attribute_mode)
        return attributes_dict

    def rate(self, preset='mazda', task_explainer=None, entities=None, attributes_dict=None, attribute_mode='compare',
             model='gpt-3.5-turbo', examples=None, n_parallel=50,
             save_folder=None, mode='base', num_runs=None, rate_first=False):
        pass
        return ratings

    def rate_texts(self, texts, preset='mazda', task_explainer=None, attributes_dict=None, attribute_mode='rate',
             model=None, examples='', n_parallel=None,
             save_folder=None, mode='rate', num_runs=None, rate_first=False, file_name = 'ratings.csv',
             attribute_category = '', object_category = '', temperature = None, timeout = None, 
             truncate_len = None, seed = None,truncate = None, use_classification = False, 
             format = None, project_probs = None, classification_clarification = None):
        
        preset_params = self.get_preset_params(preset)

        #Override the ones without None
        for param in preset_params.keys():
            if locals()[param] is None:
                locals()[param] = preset_params[param]
        
        if attribute_mode == 'rate':
            rating_function = self.attributor.rate_single_text
            call_params = {'object_category': object_category, 'attribute_category': object_category, 'api_key': self.api_key,
                      'descriptions':list(attributes_dict.values()), 'attributes': list(attributes_dict.keys()),
                      'model': preset_params['model'], 'temperature': preset_params['temperature'], 'use_classification': use_classification, 
                      'format': preset_params['format'],'project_probs': False, 'truncate': preset_params['truncate'], 
                      'seed': preset_params['seed'],'timeout': preset_params['timeout'], 'truncate_len': preset_params['truncate_len'],
                      'classification_clarification': classification_clarification}
        elif attribute_mode == 'classify':
            rating_function = generate_simple_classification
        elif attribute_mode == 'compare':
            pass

        final = pd.DataFrame()  # Initialize the final DataFrame
        processed_rows = 0  # Initialize the processed rows counter

        print('Here are the parameters for this run:\n')
        print(json.dumps(preset_params, indent=4))
        print(f'The output file will be saved at: {save_folder}/{file_name}')

        print(f'''Estimated cost: {self.estimate_cost(texts, call_params['model'])}''')
        with ThreadPoolExecutor(max_workers=preset_params['n_parallel']) as executor:
            # Pass the parameters along with the text to the rating_function
            futures = []
            for text in texts:
                # Adjust call_params for each text
                text_specific_params = call_params.copy()
                text_specific_params['text'] = text
                future = executor.submit(rating_function, **text_specific_params)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if not result.empty:
                    final = pd.concat([final, result], axis=0).reset_index(drop=True)
                processed_rows += 1


                if save_folder != None and file_name!= None:
                    final.to_csv(f'{save_folder}/{file_name}', index=False)

            return final

    def rate_with_your_prompt(self, prompt, attributes_dict=None, save_folder=None, mode='base'):
        pass
        return ratings