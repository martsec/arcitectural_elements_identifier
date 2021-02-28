"""
Architectural elements classifier library
"""

__version__ = "0.2.1"

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "The libraryn requires TensorFlow 2.0 or higher. "
        "Install TensorFlow via `pip install tensorflow`"
    ) from None
try:
    import tensorflow_hub as hub
except ImportError:
    raise ImportError(
        "The libraryn requires tensorflow_hub. "
        "Install TensorFlow hub via `pip install tensorflow_hub`"
    ) from None
from . import etl
from . import feature_eng
from . import model_def
from . import model_train
from . import model_evaluate
from . import model_deployment