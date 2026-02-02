"""CPFcluster共通型定義"""

from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

# Sentinel values
OUTLIER: int = -1
NO_PARENT: int = -1

# Type aliases
Int32Array: TypeAlias = NDArray[np.int32]
Float32Array: TypeAlias = NDArray[np.float32]
