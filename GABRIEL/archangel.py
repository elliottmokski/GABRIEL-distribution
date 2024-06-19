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
            return {'model': 'gpt-3.5-turbo', 'n_parallel': 3, 'num_runs': 10, 'rate_first': False, 'temperature':0.8,
                    'timeout': 75, 'truncate_len': 5000, 'seed': None, 'truncate':True, 'format':'json'}
        
        elif preset == 'tesla':
            return {'model': 'gpt-4-turbo', 'n_parallel': 3, 'num_runs': 10, 'rate_first': False, 'temperature':0.8,
                    'timeout': 75, 'truncate_len': 5000, 'seed': None, 'truncate':True, 'format':'json'}
        
    def get_attributes(self, attributes_dict=None, attribute_mode='compare'):
        attribute_dict = self.attributor.get_attributes(attributes_dict=attributes_dict, attribute_mode=attribute_mode)
        return attributes_dict

    # def rate(self, preset='mazda', task_explainer=None, entities=None, attributes_dict=None, attribute_mode='compare',
    #          model='gpt-3.5-turbo', examples=None, n_parallel=50,
    #          save_folder=None, mode='base', num_runs=None, rate_first=False):
    #     pass
    #     return ratings

    def rate_texts(self, texts, preset='mazda', task_description=None, attribute_dict=None, attribute_mode='rate',
             model=None, examples='', n_parallel=None,
             save_folder=None, mode='rate', num_runs=None, rate_first=False, file_name = None,
             temperature = None, timeout = None, 
             truncate_len = None, seed = None,truncate = None, use_classification = False, 
             format = None, project_probs = None, classification_clarification = None, reset_files = False,
             use_batch = False):
        
        if use_batch:
            print('''NOTE: You are using the batch API. Your request will be submitted now, and you can retrieve it in the next 24 hours when it is completed.
                  You will NOT immediately get results.''')

        if file_name == None:
            raise Exception('You must provide a file name to save to.')

        full_path = os.path.join(save_folder, file_name)
        if not os.path.isdir(save_folder):
            raise Exception('The save folder does not exist. Please provide a valid save_folder.')

        if reset_files:
            warnings.warn("Reset Files is set to True. Your files will be overwritten.")

        # Check if the file exists
        if os.path.isfile(full_path) and not reset_files and not use_batch:
            # File exists, so load the DataFrame from it
            df = pd.read_csv(full_path)
            print('File exists. DataFrame loaded from the file.')
            texts = list(df.loc[df.isna().any(axis=1)]['Text'])

        elif os.path.isfile(full_path) and use_batch:
            raise Exception('''You are using batch mode, but you have provided an existing file name, which is not supported. 
                  Provide a new, unique file name to save the batch results to.''')
        else:
            # File does not exist, create an empty DataFrame and save it
            df = pd.DataFrame(columns = ['Text'] + list(attribute_dict.keys()) + ['internal logic for gpt eyes only'])
            df['Text'] = texts
            df.to_csv(full_path, index = False)
            df = pd.read_csv(full_path)
            print(f'Creating a new file at {full_path}')
            texts = df['Text'].to_list()
        
        preset_params = self.get_preset_params(preset)
        call_params = preset_params.copy()

        for param in preset_params.keys():
            if locals()[param] == None:
                call_params[param] = preset_params[param]
            else:
                call_params[param] = locals()[param]

        call_params['project_probs'] = False
        call_params['api_key'] = self.api_key
        
        if task_description == None:
            raise Exception('Please provide a task_description')
        else:
            print(f'Extracting categories for task: {task_description}')
            categories = json.loads(identify_categories(task_description= task_description, api_key = self.api_key))
            call_params['entity_category'] = categories['entity category']
            call_params['attribute_category'] = categories['attribute category']
            call_params['classification_clarification'] = classification_clarification
            call_params['use_classification'] = use_classification

        if attribute_mode == 'rate':
            rating_function = self.attributor.rate_single_text
        elif attribute_mode == 'classify':
            rating_function = generate_simple_classification
        elif attribute_mode == 'compare':
            pass

        final = pd.DataFrame()  # Initialize the final DataFrame

        # {'model': 'gpt-3.5-turbo', 'n_parallel': 10, 'num_runs': 10, 'rate_first': False, 'temperature':0.8,
        #             'timeout': 75, 'truncate_len': 5000, 'seed': None, 'truncate':True, 'format':'json'}

        # text, attribute_dict, entity_category, attribute_category, temperature,use_classification, format, classification_clarification, 
        #                         project_probs, truncate, model, seed, api_key, truncate_len, timeout

        if use_batch:
            batch_name = file_name.split('.')[0]
            del call_params['num_runs']
            del call_params['rate_first']

            print('Here are the parameters for this run:\n')
            print(json.dumps(call_params, indent=4))
            del call_params['n_parallel']

            df['custom_id'] = df.reset_index(drop = True).index.astype(str)
            df.to_csv(full_path, index=False)
            for idx, row in df.loc[df.isna().any(axis=1)].iterrows():
                text = row['Text']
                custom_id = row['custom_id']
                path = rating_function(text = text, attribute_dict = attribute_dict, use_batch = use_batch, batch_name = batch_name, custom_id = custom_id, **call_params)
            print(f"Your raw batch information has been saved to: {path}")
            print('Beginning batch request.')
            runner = BatchRunner(self.api_key)
            batch_params = runner.run_batch(batch_name, task_description)
            batch_save_path = full_path.split('.')[0] + '_batch_metadata.csv'
            print(f'Your batch information will be saved to {batch_save_path}. Your results file at {full_path} will be populated when you retrieve the batch.')
            batch_df = create_batch_info_dataframe(batch_params)
            print(batch_df)
            batch_df.to_csv(batch_save_path)

        else:
            batches = [texts[i:i + call_params['n_parallel']] for i in range(0, len(texts),call_params['n_parallel'])]  

            del call_params['num_runs']
            del call_params['rate_first']

            print('Here are the parameters for this run:\n')
            print(json.dumps(call_params, indent=4))
            del call_params['n_parallel']

            print(f'The output file will be saved at: {save_folder}/{file_name}')
            print(f'''Rough estimated cost in dollars: {self.estimate_cost(texts, call_params['model'])}''')

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

    # def rate_with_your_prompt(self, prompt, attributes_dict=None, save_folder=None, mode='base'):
    #     pass
    #     return ratings