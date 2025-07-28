import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
os.environ.setdefault("JSON_LLM_MODEL", "dummy")
