import os
import random
from collections import OrderedDict
import json
from jinja2 import Environment, FileSystemLoader


def shuffled(it, seed=None):
    """Return a new list with the same elements, shuffled."""
    seq = list(it)
    rnd = random.Random(seed) if seed is not None else random
    rnd.shuffle(seq)
    return seq


def shuffled_dict(d, seed=None):
    """Return a JSON-formatted dict string with items shuffled."""
    items = list(d.items())
    rnd = random.Random(seed) if seed is not None else random
    rnd.shuffle(items)
    ordered = OrderedDict(items)
    return json.dumps(ordered, ensure_ascii=False, indent=2)


def get_env():
    """Return a Jinja2 environment with shuffle filters preloaded."""
    templates_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")
    env = Environment(loader=FileSystemLoader(os.path.abspath(templates_dir)))
    env.filters["shuffled"] = shuffled
    env.filters["shuffled_dict"] = shuffled_dict
    return env
