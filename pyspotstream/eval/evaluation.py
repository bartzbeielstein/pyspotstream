import gc
from time import time as get_time
import tracemalloc

import numpy as np
from sklearn import metrics as sk_metrics
from sklearn.model_selection import train_test_split
from river import stream as river_stream
from river import preprocessing as river_preprocessing
from dataclasses import dataclass
from ..misc.conventions import get_acronym


@dataclass
class ML_Evaluations:
    """
    Class for keeping track of the OML and BML evaluations.
    """
    name: str
    time: dict
    score: dict
    mem: dict
    model: object
    mini_batch_model: dict

    def __init__(self, model):
        self.name = {}
        self.time = {}
        self.score = {}
        self.model = model
        self.mini_batch_model = {}

    def eval_batch_machine_learning(
        self,
        X,
        Y,
        random_state=0,
        test_size=0.90,
        shuffle=False,
        metric=sk_metrics.mean_absolute_error,
        verbosity=True,
    ):
        """
        Standard batch machine learning experiment. The `train_test_split` method from
        `sklearn.model_selection` is applied to the input data set (`X` and `y`).

        Args:
            X (DataFrame):
                Pandas DataFrame that contains the input values.
            y (Series):
                Pandas Series that contains the output values.
            model (list):
                Batch Learner (model), e.g., `DecisionTreeRegressor` from `sklearn`.
            random_state (int): Controls the shuffling applied to the data before applying
            the split. Only needed if `shuffle=True`.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
            test_size (float or int):
                If float, should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the test split. If int, represents the
                absolute number of test samples. If None, the value is set to the
                complement of the train size. If ``train_size`` is also None, it will
                be set to 0.25.
            shuffle (bool):
                Whether or not to shuffle the data before splitting. Note: In contrast to
                `sklearn.model_selection`'s default setting (which is `True`), the default
                is `False`.
            metric: Defaults to sk_metrics.mean_absolute_error.
            verbosity (bool): If `True`, metrics are printed.

        Returns:

        """

        train_X, test_X, train_y, test_y = train_test_split(
            X, Y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )

        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        tic = get_time()
        # This evaluation uses only one model, which is the 1st in the list:
        model = self.model
        model.fit(train_X, train_y)

        model_batch_time = get_time() - tic
        model_batch_mem = tracemalloc.get_traced_memory()[1] / 1024

        pred_y = model.predict(test_X)
        model_batch_mae = metric(test_y, pred_y)

        if verbosity:
            print(
                f"{get_acronym(str(metric.__name__))} of {str(model.__class__.__name__)}: ",
                model_batch_mae,
            )
            print(f"Time of {str(model.__class__.__name__)}: ", model_batch_time)
            print(f"Memory of {str(model.__class__.__name__)}: ", model_batch_mem)

        self.time = model_batch_time
        self.score = model_batch_mae
        self.mem = model_batch_mem

    def baseline_batch_cross_val_experiment(
        self,
        X,
        Y,
        model,
        random_state=0,
        test_size=0.2,
        shuffle=False,
        metric=sk_metrics.mean_absolute_error,
        verbosity=True,
    ):
        """Not Working yet!!!.

        Args:
            X (DataFrame): Pandas DataFrame that contains the input values.
            y (Series): Pandas Series that contains the output values.
            model: Batch Learner (model), e.g., `DecisionTreeRegressor` from `sklearn`.
            random_state (int, optional): _description_. Defaults to 0.
            test_size (float, optional): _description_. Defaults to 0.2.
            shuffle (bool, optional): _description_. Defaults to False.
            metric (_type_, optional): _description_. Defaults to sk_metrics.mean_absolute_error.
            verbosity (bool, optional): _description_. Defaults to True.

        Returns:
            dict: time
            dict: score
            dict: mem
            model: model
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import make_scorer

        scoring = make_scorer(metric)
        train_X, test_X, train_y, test_y = train_test_split(
            X, Y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )

        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        tic = get_time()
        # model.fit(train_X, train_y)
        cross_val_score = cross_val_score(
            model, train_X, train_y, cv=5, scoring=scoring
        )
        model_batch_time = get_time() - tic
        model_batch_mem = tracemalloc.get_traced_memory()[1] / 1024
        #
        mean_cross_val_score = cross_val_score.mean()
        # or !
        # cross_validate
        #
        pred_y = model.predict(test_X)
        model_batch_mae = metric(test_y, pred_y)

        if verbosity:
            print(f"Cross val. score of {model}: ", mean_cross_val_score)
            print(f"MAE of {model}: ", model_batch_mae)
            print(f"Time of {model}: ", model_batch_time)
            print(f"Memory of {model}: ", model_batch_mem)

        return model_batch_time, mean_cross_val_score, model_batch_mem, model

    def eval_mini_batch_machine_learning(
        self,
        X,
        y,
        x_part="linspace",
        m_sklearn=True,
        metric=sk_metrics.mean_absolute_error,
        eval_on_full_data=False,
        fit_on_available_data=False,
        fit_on_fixed=False,
        fixed_train_size=0,
        n_fit=10,
        n_splits=100,
        # return_as_df = False
        verbose=False,
    ):
        """This method executes a mini-batch experiment.

        Args:
            X (DataFrame): Pandas DataFrame that contains the input values.
            y (Series): Pandas Series that contains the output values.
            model: Batch Learner (model), e.g., `DecisionTreeRegressor` from `sklearn`.
            x_part (str, optional):
                Partition of the input space. Defaults to 'linspace'.
            m_sklearn (bool, optional):
                Indicator for `sklearn`models. If `False`, then `river` models are used.
            metric:
                Evaluation metric. Defaults to sk_metrics.mean_absolute_error.
            eval_on_full_data (bool, optional):
                Evaluation uses the full data set. Defaults to False.
            fit_on_available_data (bool, optional):
                Fit the model on the whole set of observed data.
            fit_on_fixed (bool, optional):
                Fit the model on a fixed data set. Defaults to False.
            fixed_train_size (int, optional):
                Size of the fixed data set. Defaults to 0.
            n_fit (int, optional):
                Fit (train) on the last n_fit partitions only.
                This is a moving window of size (n_fit x partition size, which is computed via `x_part`).
                Defaults to 10.
            n_splits (int, optional):
                Number of splits (mini-batches) used for `x_part`. Defaults to 100.
            verbose (bool, optional):
                verbosity level. Defaults to False.

        Returns:

        """
        start_time = get_time()
        model = self.model
        time = {}
        score = {}
        mem = {}

        if x_part == "logspace":
            x_seq = np.logspace(
                0.5, 1, n_splits, base=X.shape[0], dtype=int, endpoint=eval_on_full_data
            )[1:]
        elif x_part == "linspace":
            x_seq = np.linspace(
                1, X.shape[0], n_splits, dtype=int, endpoint=eval_on_full_data
            )[1:]
        else:
            raise Exception(f"{x_part} is an invalid Argument for x_part")

        if verbose:
            print("Length of Trainings Data", X.shape[0])
        if verbose:
            print(f"x_seq: {x_seq}")

        model_dict = {}
        tracemalloc.start()

        if m_sklearn:
            # Sklearn Model: Start of Train and Eval Loop
            if verbose:
                print("Starting Loop for Sklearn Model")
            for i, break_point in enumerate(x_seq):

                model_dict[break_point] = model
                if verbose:
                    print(f"{i}. Breaking Point: {break_point}")

                gc.collect()
                tracemalloc.reset_peak()
                tic = get_time()

                # -- TRAINING --
                if fit_on_fixed:
                    # train on a fixed data set of size 0:fixed_train_size
                    if verbose:
                        print(f"\tFit on 0:{fixed_train_size}.")
                    model_dict[break_point].fit(
                        X.iloc[:fixed_train_size], y.iloc[:fixed_train_size]
                    )

                elif fit_on_available_data or i < n_fit:
                    # train on the full set of seen data,
                    # that is available until train_size, i.e., from 0 to train_size:
                    if verbose:
                        print(f"\tFit on 0:{break_point}.")
                    model_dict[break_point].fit(
                        X.iloc[:break_point], y.iloc[:break_point]
                    )

                else:
                    # train on the last n_fit partitions only. This is a moving window of size (n_fit x partition size).
                    if verbose:
                        print(f"\tFit on {x_seq[i - n_fit]}:{break_point}.")
                    model_dict[break_point].fit(
                        X.iloc[x_seq[i - n_fit]:break_point],
                        y.iloc[x_seq[i - n_fit]:break_point],
                    )

                # -- EVALUATION --
                if eval_on_full_data:
                    # predict and evaluate on the full data set:
                    if verbose:
                        print("\tPredict on full X.")
                    y_pred = model_dict[break_point].predict(X)  # data leakage!!!!
                    score[break_point] = metric(y, y_pred)

                elif break_point != x_seq[-1]:
                    # predict and evaluate on the next sequence
                    next_break_point = x_seq[i + 1]
                    if verbose:
                        print(f"\tPredict on {break_point}:{next_break_point}.")
                    y_pred = model_dict[break_point].predict(
                        X.iloc[break_point:next_break_point]
                    )  # data leakage!!!!
                    score[break_point] = metric(
                        y.iloc[break_point:next_break_point], y_pred
                    )
                else:
                    if verbose:
                        print("\tNo more Data to predict on!")

                time[break_point] = get_time() - tic
                mem[break_point] = tracemalloc.get_traced_memory()[1] / 1024

            tracemalloc.stop()
            self.time = time
            self.score = score
            self.mem = mem
            self.mini_batch_model = model_dict
            runtime = np.round(get_time() - start_time, 2)
            print(f"Completed after {runtime} seconds.")

        else:
            # River Model: Start of Train and Eval Loop
            if verbose:
                print("Starting Loop for River Model")
            for i, break_point in enumerate(x_seq):

                model_dict[break_point] = model
                if verbose:
                    print(f"{i}. Breaking Point: {break_point}")

                gc.collect()
                tracemalloc.reset_peak()
                tic = get_time()

                # -- TRAINING --
                if fit_on_fixed:
                    # train on a fixed data set of size 0:fixed_train_size
                    if verbose:
                        print(f"\tFit on 0:{fixed_train_size}.")
                    model_dict[break_point].learn_many(
                        X.iloc[:fixed_train_size], y.iloc[:fixed_train_size]
                    )

                elif fit_on_available_data or i < n_fit:
                    # train on the full set of seen data,
                    # that is available until train_size, i.e., from 0 to train_size:
                    if verbose:
                        print(f"\tFit on 0:{break_point}.")
                    model_dict[break_point].learn_many(
                        X.iloc[:break_point], y.iloc[:break_point]
                    )

                else:
                    # train on the last n_fit partitions only.
                    # This is a moving window of size (n_fit x partition size).
                    if verbose:
                        print(f"\tFit on {x_seq[i - n_fit]}:{break_point}.")
                    model_dict[break_point].learn_many(
                        X.iloc[x_seq[i - n_fit]:break_point],
                        y.iloc[x_seq[i - n_fit]:break_point],
                    )

                # -- EVALUATION --
                if eval_on_full_data:
                    # predict and evaluate on the full data set:
                    if verbose:
                        print("\tPredict on full X.")
                    y_pred = model_dict[break_point].predict_many(X)  # data leakage!!!!
                    score[break_point] = metric(y, y_pred)

                elif break_point != x_seq[-1]:
                    # predict and evaluate on the next sequence
                    next_break_point = x_seq[i + 1]
                    if verbose:
                        print(f"\tPredict on {break_point}:{next_break_point}.")
                    y_pred = model_dict[break_point].predict_many(
                        X.iloc[break_point:next_break_point]
                    )  # data leakage!!!!
                    score[break_point] = metric(
                        y.iloc[break_point:next_break_point], y_pred
                    )
                else:
                    if verbose:
                        print("\tNo more Data to predict on!")

                time[break_point] = get_time() - tic
                mem[break_point] = tracemalloc.get_traced_memory()[1] / 1024

            tracemalloc.stop()
            self.time = time
            self.score = score
            self.mem = mem
            runtime = np.round(get_time() - start_time, 2)
            print(f"Completed after {runtime} seconds.")

    def eval_online_machine_learning(self, X, y, metric, task="clf"):
        """
        This methods executes an online machine learning task
        (every single instance is processed separately).

        Args:
            X (DataFrame): _description_
            y (DataFrame): _description_
            metric (_type_): metric
            task (str, optional): task_description_. Defaults to "clf".

        """
        start_time = get_time()
        model = self.model
        time = {}
        score = {}
        mem = {}
        scaler = river_preprocessing.StandardScaler()
        i = 0
        y_true = []
        y_pred = []

        tracemalloc.start()
        gc.collect()
        for xi, yi in river_stream.iter_pandas(X, y):
            tic = get_time()

            xi_scaled = scaler.learn_one(xi).transform_one(xi)
            yi_pred = model.predict_one(xi_scaled)

            tracemalloc.reset_peak()

            model.learn_one(xi_scaled, yi)
            mem[i] = tracemalloc.get_traced_memory()[1] / 1024
            time[i] = get_time() - tic

            if task == "clf":
                y_pred.append(bool(y_pred))
                y_true.append(bool(yi))
            elif task == "reg":
                y_pred.append(yi_pred)
                y_true.append(yi)
            else:
                raise Exception(f"{task} is an invalid input for task")

            score[i] = metric(y_true, y_pred)
            i = i + 1

        tracemalloc.stop()
        self.time = time
        self.score = score
        self.mem = mem
        runtime = np.round(get_time() - start_time, 2)
        print(f"Completed after {runtime} seconds.")
