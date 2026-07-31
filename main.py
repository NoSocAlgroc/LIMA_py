from LIMA.LIMA import LIMA
from LIMA.sampler.LIMASampler import LIMASampler
from LIMA.scheduler.LIMAScheduler import LIMAScheduler
FILE_PATH = "flights.csv"  # set to the dataset CSV to run LIMA on
N = 100000  # samples per step
MAX_ITERATIONS = 20


def main() -> None:
    scheduler = LIMAScheduler()
    sampler = LIMASampler()

    lima = LIMA(FILE_PATH, scheduler=scheduler, sampler=sampler)
    for i in range(MAX_ITERATIONS):
        done = lima.step(N)
        if done:
            print(f"Stopped early at iteration {i}: schedule empty.")
            return
    print(f"Ran {MAX_ITERATIONS} iterations without an empty schedule.")


if __name__ == "__main__":
    main()
