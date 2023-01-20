import matplotlib.pyplot as plt


def plot_multiple_instances_results(
    alg_dict,
    y_label,
    marker=".",
    alpha=0.7,
    default=None,
    log_x=False,
    log_y=False,
    title="",
):

    fig, ax = plt.subplots(figsize=(14, 8))

    if default is not None:
        ax.axhline(y=default, color="black", linestyle="-.", label="default")

    for i, (algo_k, algo_v) in enumerate(alg_dict.items()):
        ax.plot(
            algo_v.keys(),
            algo_v.values(),
            alpha=alpha,
            marker=marker,
            linestyle="dashed",
            label=algo_k,
        )

    ax.set_title = title
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_ylim(0)
    ax.grid(True)
    ax.set_xlabel("Number of train observations")
    ax.set_ylabel(y_label)
    ax.legend()
    plt.show()
