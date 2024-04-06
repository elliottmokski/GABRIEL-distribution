from gabriel.foundational_functions import *
from gabriel.combined_assistant import CombinedAssistant
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import json
import os
import warnings

class Archangel():
    def __init__(self, api_key) -> None:
        self.attributor = CombinedAssistant(api_key)
        self.api_key = api_key

    def estimate_cost(self, texts, model):
        if 'gpt-3.5' in model:
            return round(sum([(len(text.split()) + 3000 + 2000) for text in texts]) / 1000 / 1000 * 1.33 * 0.5, 3)
        elif 'gpt-4' in model:
            return round(sum([(len(text.split()) + 3000 + 2000) for text in texts]) / 1000 / 1000 * 1.33 * 20,3)
        else:
            return 'Unknown model'

    def get_preset_params(self, preset='mazda'):
        if preset == 'mazda':
            return {'model': 'gpt-3.5-turbo', 'n_parallel': 50, 'num_runs': 10, 'rate_first': False, 'temperature':0.8,
                    'timeout': 75, 'truncate_len': 5000, 'seed': None, 'truncate':True, 'format':'json',"batch_len":50}
        
        elif preset == 'tesla':
            return {'model': 'gpt-4-turbo', 'n_parallel': 30, 'num_runs': 10, 'rate_first': False, 'temperature':0.8,
                    'timeout': 75, 'truncate_len': 5000, 'seed': None, 'truncate':True, 'format':'json', "batch_len":50}
        
    def get_attributes(self, attributes_dict=None, attribute_mode='compare'):
        attribute_dict = self.attributor.get_attributes(attributes_dict=attributes_dict, attribute_mode=attribute_mode)
        return attributes_dict

    def rate(self, preset='mazda', task_explainer=None, entities=None, attributes_dict=None, attribute_mode='compare',
             model='gpt-3.5-turbo', examples=None, n_parallel=50,
             save_folder=None, mode='base', num_runs=None, rate_first=False):
        pass
        return ratings

    def rate_texts(self, texts, preset='mazda', task_description=None, attribute_dict=None, attribute_mode='rate',
             model=None, examples='', n_parallel=None,
             save_folder=None, mode='rate', num_runs=None, rate_first=False, file_name = None,
             temperature = None, timeout = None, 
             truncate_len = None, seed = None,truncate = None, use_classification = False, 
             format = None, project_probs = None, classification_clarification = None, batch_len = None, reset_files = False):
        
        if file_name == None:
            raise Exception('You must provide a file name to save to.')

        full_path = os.path.join(save_folder, file_name)
        if not os.path.isdir(save_folder):
            raise Exception('The save folder does not exist. Please provide a valid save_folder.')

        if reset_files:
            warnings.warn("Reset Files is set to True. Your files will be overwritten.")

        # Check if the file exists
        if os.path.isfile(full_path) and not reset_files:
            # File exists, so load the DataFrame from it
            df = pd.read_csv(full_path)
            print('File exists. DataFrame loaded from the file.')
            texts = list(df.loc[df.isna().any(axis=1)]['Text'])
        else:
            # File does not exist, create an empty DataFrame and save it
            df = pd.DataFrame(columns = ['Text'] + list(attribute_dict.keys()) + ['internal logic for gpt eyes only'])
            df['Text'] = texts
            df.to_csv(full_path, index = False)
            df = pd.read_csv(full_path)
            print(f'Creating a new file at {full_path}')
            texts = df['Text'].to_list()
        
        preset_params = self.get_preset_params(preset)

        #Override the ones without None
        for param in preset_params.keys():
            if locals()[param] is None:
                locals()[param] = preset_params[param]
        
        if task_description == None:
            raise Exception('Please provide a task_description')
        else:
            print(f'Extracting categories for task: {task_description}')
            categories = json.loads(identify_categories(task_description= task_description, api_key = self.api_key))
            entity_category = categories['entity category']
            attribute_category = categories['attribute category']

        if attribute_mode == 'rate':
            rating_function = self.attributor.rate_single_text
            call_params = {'entity_category': entity_category, 'attribute_category': attribute_category, 'api_key': self.api_key,
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
        print(f'''Rough estimated cost in dollars: {self.estimate_cost(texts, call_params['model'])}''')

        batches = [texts[i:i + preset_params['batch_len']] for i in range(0, len(texts), preset_params['batch_len'])]  
        
        for batch in tqdm(batches, total=len(batches)):
            batch_results = []
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {executor.submit(rating_function, text = text, attribute_dict = attribute_dict, **call_params): text for text in batch}
                for future in as_completed(futures):
                    text = futures[future]
                    try:
                        ranking_results = future.result()
                        if ranking_results is not None:
                            # Store the results for each group to process later
                            batch_results.append(ranking_results)
                    except Exception as exc:
                        print(f"Generated an exception for text: {text[:100]}: {exc}")
            curr_df = pd.concat(batch_results, axis=0).reset_index(drop=True)
            df, curr_df = update_dataframe(df, curr_df, list(attribute_dict.keys()) + ['internal logic for gpt eyes only'])
            df.to_csv(full_path, index=False)

        return df
        # with ThreadPoolExecutor(max_workers=preset_params['n_parallel']) as executor:
        #     # Pass the parameters along with the text to the rating_function
        #     futures = []
        #     for text in texts:
        #         # Adjust call_params for each text
        #         text_specific_params = call_params.copy()
        #         text_specific_params['text'] = text
        #         future = executor.submit(rating_function, attribute_dict = attribute_dict, **text_specific_params)
        #         futures.append(future)

        #     for future in tqdm(as_completed(futures), total=len(futures)):
        #         result = future.result()
        #         if not result.empty:
        #             final = pd.concat([final, result], axis=0).reset_index(drop=True)
        #         processed_rows += 1

        #         final.to_csv(f'{save_folder}/{file_name}', index=False)

        #     return final

    def rate_with_your_prompt(self, prompt, attributes_dict=None, save_folder=None, mode='base'):
        pass
        return ratings