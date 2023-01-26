import gc
import time
import tracemalloc
import copy

from river import stream as river_stream
from dataclasses import dataclass, field


@dataclass
class ExperimentResults:
    """
    Class for keeping track of the OML and BML evaluations.
    """
    name: str = None
    score: dict = field(default_factory=dict)
    time: dict = field(default_factory=dict)
    mem: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)

    def plot_results(self):
        pass


@dataclass
class Tracer():
    end_time: float = None
    end_mem: float = None
    _start_time: float = None
    _mem_tracer: object = tracemalloc

    def start(self) -> None:
        gc.collect()
        self._mem_tracer.start()
        self._mem_tracer.reset_peak()
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        # Calculate end time
        self.end_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Calculate end mem
        self.end_mem = self._mem_tracer.get_traced_memory()[1] / 1024
        return self.end_time, self.end_mem

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()


def run_batch_ml(x_train, x_test, y_train, y_test, model, metric):
    with Tracer() as t:
        model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    score = metric(y_test, y_pred)

    return ExperimentResults(name=str(model.__class__.__name__),
                             score={0: score},
                             time={0: t.end_time},
                             mem={0: t.end_mem},
                             model={0: model})


def run_mini_batch_sklearn(X,
                           y,
                           x_seq,
                           model,
                           metric,
                           eval_on_full_data=False,
                           fit_on_available_data=False,
                           fit_on_fixed=False,
                           fixed_train_size=0,
                           n_fit=10,
                           ):
    res = ExperimentResults(name=str(model.__class__.__name__))

    for i, break_point in enumerate(x_seq):

        res.model[break_point] = copy.deepcopy(model)

        print(f"{i}. Breaking Point: {break_point}")

        with Tracer() as t:

            # -- TRAINING --
            if fit_on_fixed:
                # train on a fixed data set of size 0:fixed_train_size

                print(f"\tFit on 0:{fixed_train_size}.")
                res.model[break_point].fit(X.iloc[:fixed_train_size], y.iloc[:fixed_train_size])

            elif fit_on_available_data or i < n_fit:
                # train on the full set of seen data,
                # that is available until train_size, i.e., from 0 to train_size:

                print(f"\tFit on 0:{break_point}.")
                res.model[break_point].fit(X.iloc[:break_point], y.iloc[:break_point])

            else:
                # train on the last n_fit partitions only. This is a moving window of size (n_fit x partition size).

                print(f"\tFit on {x_seq[i - n_fit]}:{break_point}.")
                res.model[break_point].fit(X.iloc[x_seq[i - n_fit]: break_point],
                                           y.iloc[x_seq[i - n_fit]: break_point], )

            # -- EVALUATION --
            if eval_on_full_data:
                # predict and evaluate on the full data set:

                print("\tPredict on full X.")
                y_pred = res.model[break_point].predict(X)  # data leakage!!!!
                res.score[break_point] = metric(y, y_pred)

            elif break_point != x_seq[-1]:

                # predict and evaluate on the next sequence
                next_break_point = x_seq[i + 1]

                print(f"\tPredict on {break_point}:{next_break_point}.")

                y_pred = res.model[break_point].predict(X.iloc[break_point:next_break_point])  # data leakage!!!!
                res.score[break_point] = metric(y.iloc[break_point:next_break_point], y_pred)

            else:
                print("\tNo more Data to predict on!")

        res.time[break_point] = t.end_time
        res.mem[break_point] = t.end_mem

    return res


def run_mini_batch_river(X,
                         y,
                         x_seq,
                         model,
                         metric,
                         eval_on_full_data=False,
                         fit_on_available_data=False,
                         fit_on_fixed=False,
                         fixed_train_size=0,
                         n_fit=10,
                         ):
    res = ExperimentResults(name=str(model.__class__.__name__))

    for i, break_point in enumerate(x_seq):

        res.model[break_point] = copy.deepcopy(model)

        print(f"{i}. Breaking Point: {break_point}")

        with Tracer() as t:
            # -- TRAINING --
            if fit_on_fixed:
                # train on a fixed data set of size 0:fixed_train_size
                print(f"\tFit on 0:{fixed_train_size}.")
                res.model[break_point].learn_many(
                    X.iloc[:fixed_train_size], y.iloc[:fixed_train_size]
                )

            elif fit_on_available_data or i < n_fit:
                # train on the full set of seen data,
                # that is available until train_size, i.e., from 0 to train_size:
                print(f"\tFit on 0:{break_point}.")
                res.model[break_point].learn_many(
                    X.iloc[:break_point], y.iloc[:break_point]
                )

            else:
                # train on the last n_fit partitions only.
                # This is a moving window of size (n_fit x partition size).
                print(f"\tFit on {x_seq[i - n_fit]}:{break_point}.")
                res.model[break_point].learn_many(
                    X.iloc[x_seq[i - n_fit]: break_point],
                    y.iloc[x_seq[i - n_fit]: break_point],
                )

            # -- EVALUATION --
            if eval_on_full_data:
                # predict and evaluate on the full data set:
                print("\tPredict on full X.")
                y_pred = res.model[break_point].predict_many(X)  # data leakage!!!!
                res.score[break_point] = metric(y, y_pred)

            elif break_point != x_seq[-1]:
                # predict and evaluate on the next sequence
                next_break_point = x_seq[i + 1]
                print(f"\tPredict on {break_point}:{next_break_point}.")
                y_pred = res.model[break_point].predict_many(
                    X.iloc[break_point:next_break_point]
                )  # data leakage!!!!
                res.score[break_point] = metric(
                    y.iloc[break_point:next_break_point], y_pred
                )
            else:
                print("\tNo more Data to predict on!")

        res.time[break_point] = t.end_time
        res.mem[break_point] = t.end_mem

    return res


def run_online_ml(X, y, model, metric):
    y_true = []
    y_pred = []

    res = ExperimentResults(name=str(model.__class__.__name__))

    for i, (xi, yi) in enumerate(river_stream.iter_pandas(X, y)):
        yi_pred = model.predict_one(xi)
        with Tracer() as t:
            model.learn_one(xi, yi)

        y_pred.append(yi_pred)
        y_true.append(yi)

        res.score[i] = metric(y_true, y_pred)
        res.time[i] = t.end_time
        res.mem[i] = t.end_mem

    res.model = [model]

    return res
