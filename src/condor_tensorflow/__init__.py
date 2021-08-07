from .version import __version__

from .loss import CondorOrdinalCrossEntropy
from .loss import SparseCondorOrdinalCrossEntropy
from .metrics import MeanAbsoluteErrorLabels
from .metrics import EarthMoversDistanceLabels
from .activations import ordinal_softmax
from .labelencoder import CondorOrdinalEncoder

# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
  'MeanAbsoluteErrorLabels',
  'EarthMoversDistanceLabels',
  'CondorOrdinalCrossEntropy',
  'SparseCondorOrdinalCrossEntropy',
  'ordinal_softmax',
  'CondorOrdinalEncoder',
]
