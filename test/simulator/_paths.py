import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
eventQueue_dir = os.path.join(current_dir, "../../")
sys.path.append(eventQueue_dir)
