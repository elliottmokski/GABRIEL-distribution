import sys
from pathlib import Path
import numpy as np

# Add the parent directory of Gabriel_2 to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Gabriel_2.foundational_functions import *
from Gabriel_2.combined_assistant import CombinedAssistant

passages = ['She was ecstatic about the Rhodes fellowship','She was distraught.','He was satisfied with his performance on the exam, but unhappy with how his friend did.', np.nan]
attributes = ['happy, sad, angry, surprised, grateful']
descriptions = ['The happy attribute refers to a positive emotional state characterized by feelings of joy, contentment, and satisfaction.', 'The "sad" attribute is a feeling of sorrow, unhappiness, or distress. It is an emotional state characterized by a low mood and a sense of loss or disappointment.', "The angry attribute refers to a strong feeling of displeasure or hostility, often accompanied by a desire to retaliate or express one's frustration.", 'The surprised attribute refers to the feeling of astonishment or amazement resulting from unexpected events or information. It is characterized by a sudden shift in mental and emotional state, often accompanied by widened eyes, raised eyebrows, and an open mouth.', 'Grateful is the feeling of being thankful and appreciative for something or someone, acknowledging the good things in life.']


CombinedAssistant('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').simple_evaluation_pipeline(search_axis_1 = passages, attribute_category = 'emotions',
                                                  object_category = 'short stories', seed = 0, attributes = attributes, descriptions= descriptions, format = 'json', use_classification = True)

# print(generate_simple_classification(attributes = attributes, descriptions = descriptions,
                            #    passage = passages[0], object_category= 'short stories', attribute_category = 'emotions', seed = 0))

# passage = passages[0]
# object_category = 'short stories'
# attribute_category = 'emotions'

# print(generate_simple_ratings(attributes = attributes, descriptions = descriptions,
#                                                 passage = passage, object_category= object_category, 
#                                                 attribute_category=attribute_category,format = 'json', seed = 1))

# generate_simple_ratings(attributes, descriptions, passage,
#                             object_category, attribute_category,format = 'json',
#                             timeout=90, temperature=0.8, model='gpt-3.5-turbo-1106', seed = 1)

# category = 'vegetables'

# template_context = {
#     'n_items': 7,  # Specify the number of items you want
#     'category': category,  # Specify the category
#     'object_clarification': '',  # Any additional clarification if needed
#     'attribute_clarification': '',  # Any additional clarification if needed
#     'mode': 'attribute'  # 'item' or 'attribute' depending on what you are generating
# }

# # Call the function with the context
# items = generate_category_items(timeout=90, temperature=0.8, model='gpt-3.5-turbo-1106', template_context=template_context)

# descriptions = get_attribute_descriptions(category, items)
# print(descriptions)