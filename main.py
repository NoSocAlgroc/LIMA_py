import numpy as np
import numpy.typing as npt

from LIMA.LIMA import LIMA
from LIMA.sampler.LIMASampler import LIMASampler
from LIMA.scheduler.LIMAScheduler import LIMAScheduler


#Safety bound, never reached.
MAX_ITERATIONS = 100


def runLima(dataset_path: str, approx: float) -> npt.NDArray[np.float64]:
    """Run LIMA to completion on `dataset_path` with the given approximation
    factor, then return its attribute relatedness matrix."""
    lima = LIMA(
        dataset_path,
        scheduler=LIMAScheduler(),
        sampler=LIMASampler(),
        approx=approx,
    )
    for i in range(MAX_ITERATIONS):
        if lima.step(i):
            break
    return lima.relatedness()


if __name__ == "__main__":
    matrix = runLima("jannis.csv", 0.000001)
    np.savetxt("output.txt", matrix, fmt="%.2f")
