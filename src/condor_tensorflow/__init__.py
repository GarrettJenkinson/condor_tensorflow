from .version import __version__

from .loss import CondorOrdinalCrossEntropy
from .loss import SparseCondorOrdinalCrossEntropy
from .loss import OrdinalEarthMoversDistance
from .loss import SparseOrdinalEarthMoversDistance
from .metrics import OrdinalMeanAbsoluteError
from .metrics import SparseOrdinalMeanAbsoluteError
#from .metrics import OrdinalEarthMoversDistance
#from .metrics import SparseOrdinalEarthMoversDistance
from .activations import ordinal_softmax
from .labelencoder import CondorOrdinalEncoder

# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
    'OrdinalMeanAbsoluteError',
    'SparseOrdinalMeanAbsoluteError',
    'OrdinalEarthMoversDistance',
    'SparseOrdinalEarthMoversDistance',
    'CondorOrdinalCrossEntropy',
    'SparseCondorOrdinalCrossEntropy',
    'ordinal_softmax',
    'CondorOrdinalEncoder',
]
