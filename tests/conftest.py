# conftest.py
import sys, os
# adjust this to wherever your top-level package lives
project_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, project_src)
