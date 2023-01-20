import time
import datetime as dt
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import cross_val_score
from river import evaluate


class plotPerformance:
    """Basic plot with three subplots: error, time, and memory"""

    def __init__(self, start=0, end=None, figsize=(10, 5)):
        self.figsize = figsize
        self.nrows = 3
        self.dpi = 300
        self.start = start
        self.end = end

    def plot_performance(self, dataset, metric, models):
        metric_name = metric.__class__.__name__

        # To make the generated data reusable
        dataset = list(dataset)[self.start : self.end]
        fig, ax = plt.subplots(figsize=self.figsize, nrows=self.nrows, dpi=self.dpi)
        for model_name, model in models.items():
            step = []
            error = []
            r_time = []
            memory = []

            for checkpoint in evaluate.iter_progressive_val_score(
                dataset, model, metric, measure_time=True, measure_memory=True, step=100
            ):
                step.append(checkpoint["Step"])
                error.append(checkpoint[metric_name].get())

                # Convert timedelta object into seconds
                r_time.append(checkpoint["Time"].total_seconds())
                # Make sure the memory measurements are in MB
                raw_memory = checkpoint["Memory"]
                memory.append(raw_memory * 2**-20)

            ax[0].plot(step, error, label=model_name)
            ax[1].plot(step, r_time, label=model_name)
            ax[2].plot(step, memory, label=model_name)

        ax[0].set_ylabel(metric_name)
        ax[1].set_ylabel("Time (seconds)")
        ax[2].set_ylabel("Memory (MB)")
        ax[2].set_xlabel("Instances")

        ax[0].grid(True)
        ax[1].grid(True)
        ax[2].grid(True)

        ax[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.25),
            ncol=3,
            fancybox=True,
            shadow=True,
        )
        plt.tight_layout()
        plt.close()

        return fig


def model_memory_eval(obj, unit="byte", protocol=3):
    """
    Model size estimation with pickle
    Parameters
    ----------
    obj: python object

    unit : str, default='byte'
        'kB'
        'MB'

    protocol : int, default=3
    """
    import pickle
    import sys

    size = sys.getsizeof(pickle.dumps(obj, protocol=protocol))
    # print('pickle size:',size , 'byte')
    if unit == "kB":
        return size / 1024

    if unit == "MB":
        return size / (2**20)

    return size


def batch_tree_performance(x, y):
    """Estimate performance (error, time, and memory usage) for a Decision Tree Classifier from the sklearn package"""

    # To evaluate time and memory usage, we train a Decision Tree Classifier (DTC) on the entire dataset.
    time_start = time.perf_counter()
    dtc = tree.DecisionTreeClassifier()
    dtc = dtc.fit(x, y)
    time_now = time.perf_counter()
    time_delta = dt.timedelta(seconds=time_now - time_start)
    memory_usage = model_memory_eval(dtc)

    # Evaluate a model with 5-fold cross-validation
    scores = cross_val_score(tree.DecisionTreeClassifier(), x, y, cv=5)
    mean_score = scores.mean()

    # Return time, memory usage and model error
    return time_delta.microseconds / 1000, memory_usage / 1024, mean_score
