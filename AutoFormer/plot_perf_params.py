import argparse
import pathlib
import typing
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams

rcParams['font.size'] = 22
rcParams['font.weight'] = 'bold'
rcParams['legend.fontsize'] = 22
plt.style.use('ggplot')

Metric = namedtuple("Metric", "col_name full_name plot_low plot_high")

TASK_TO_METRIC = {
    'cifar100': Metric("test_acc_1", "Accuracy@1", 63, 71),
    'swinir': Metric("psnr", "PSNR", 23.5, 25),
    'roberta': Metric("ppl", "MLM Perplexity", 14, 26),
}


def plot_all(evolution_input_path: typing.Union[str, pathlib.Path],
             random_search_input_path: typing.Union[str, pathlib.Path],
             output_path: typing.Union[str, pathlib.Path],
             metric: Metric,
             tag: str = None):
    output_path = pathlib.Path(output_path)
    tag = f'{tag}_' if tag is not None else ''

    ev_data_df = pd.read_json(evolution_input_path, lines=True)
    rs_data_df = pd.read_json(random_search_input_path, lines=True)

    OPACITY = 0.3

    def generate_scatter_plot(ev_data_df, rs_data_df, x_col, x_label, perf_col):
        fig, ax = plt.subplots(figsize=(14, 8))

        ax.set_xlabel(x_label, fontsize=22, color='black', fontweight='bold')
        ax.set_ylabel(metric.full_name, fontsize=22, color='black', fontweight='bold')
        ax.set_ylim(metric.plot_low, metric.plot_high)

        mix_scatter = True
        if not mix_scatter:
            ax.scatter(ev_data_df[x_col], ev_data_df[perf_col], alpha=OPACITY, c='tab:red', label='Evolution')
            ax.scatter(rs_data_df[x_col], rs_data_df[perf_col], alpha=OPACITY, c='tab:blue', label='Random')
        else:
            all_x = np.concatenate([ev_data_df[x_col].values, rs_data_df[x_col].values])
            all_y = np.concatenate([ev_data_df[perf_col].values, rs_data_df[perf_col].values])
            np.random.seed(42)
            colors = np.array(['tab:red'] * len(ev_data_df) + ['tab:blue'] * len(rs_data_df))
            random_order = np.random.choice(len(all_x), len(all_x), replace=False)
            ax.scatter(all_x[random_order], all_y[random_order], alpha=OPACITY, c=colors[random_order])

            # Phantom plots just to register the legends
            ax.scatter([], [], color='tab:red', label='Evolution')
            ax.scatter([], [], color='tab:blue', label='Random')

        leg = plt.legend(markerscale=1.5)
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        # Have the ticks be integers
        plt.locator_params(axis="both", integer=True, tight=True)

        return fig, ax

    ## Epoch vs Performance
    fig, ax = generate_scatter_plot(ev_data_df, rs_data_df,
                                    x_col='epoch', x_label='Search Iteration',
                                    perf_col=metric.col_name)
    # plt.tight_layout()
    fig.savefig(output_path / f'{tag}perf_for_top_50_every_epoch.png', dpi=450, bbox_inches='tight')
    plt.show()

    ## Param-size vs Performance
    fig, ax = generate_scatter_plot(ev_data_df, rs_data_df,
                                    x_col='params', x_label='Number of Parameters (in Millions)',
                                    perf_col=metric.col_name)
    plt.savefig(output_path / f'{tag}perf_vs_params_for_highperf_cand.png', dpi=450, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder',
                        type=str, nargs='?',
                        help='path of folder with evolution and random search results')
    parser.add_argument('--input-path',
                        type=str, nargs='?',
                        help='path of file with random search results')
    parser.add_argument('--input-path-random',
                        type=str, nargs='?',
                        help='path of file with random search results')
    parser.add_argument('--result_dir', type=str, nargs='?',
                        help='name of folder where plot files will be dumped')
    parser.add_argument('--task', type=str, nargs='?',
                        help='The identifier for the task. One of (cifar100, roberta, swinir)')
    args = parser.parse_args()

    task = args.task
    input_path_evolution = args.input_path
    if input_path_evolution is None:
        input_path_evolution = pathlib.Path(args.input_folder) / f'{task}_search.json'

    input_path_random = args.input_path
    if input_path_random is None:
        input_path_random = pathlib.Path(args.input_folder) / f'{task}_search_random.json'

    result_dir = args.result_dir
    if result_dir is None:
        result_dir = args.input_folder

    metric = TASK_TO_METRIC[task]

    plot_all(
        input_path_evolution,
        input_path_random,
        result_dir,
        metric,
        tag=task
    )


if __name__ == '__main__':
    main()
