import argparse
import pathlib
import typing

import matplotlib.pyplot as plt
from pylab import rcParams
import pandas as pd

rcParams['font.size'] = 16
rcParams['legend.fontsize'] = 20
plt.style.use('ggplot')

METRIC_PLOT_RANGE = {
    'ppl': (12, 30)
}


def plot_all(evolution_input_path: typing.Union[str, pathlib.Path], 
             random_search_input_path: typing.Union[str, pathlib.Path],
             output_path: typing.Union[str, pathlib.Path],
             tag: str = None):

    output_path = pathlib.Path(output_path)
    tag = f'_{tag}' if tag is not None else ''

    ev_data_df = pd.read_json(evolution_input_path, lines=True)
    rs_data_df = pd.read_json(random_search_input_path, lines=True)

    perf_col = 'ppl'
    perf_metric = 'MLM Perplexity'
    OPACITY = 0.3

    def generate_scatter_plot(ev_data_df, rs_data_df, x_col, x_label, perf_col):
        fig, ax = plt.subplots(figsize=(14, 8))

        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(perf_metric, fontsize=16)
        if perf_col in METRIC_PLOT_RANGE:
            ax.set_ylim(*METRIC_PLOT_RANGE[perf_col])

        ax.scatter(ev_data_df[x_col], ev_data_df[perf_col], alpha=OPACITY, c='tab:red', label='Evolution')
        ax.scatter(rs_data_df[x_col], rs_data_df[perf_col], alpha=OPACITY, c='tab:blue', label='Random')

        # Have the ticks be integers
        plt.locator_params(axis="both", integer=True, tight=True)

        leg = plt.legend(markerscale=1.5)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)

        return fig, ax

    ## Epoch vs Performance
    fig, ax = generate_scatter_plot(ev_data_df, rs_data_df, 
                                    x_col='epoch', x_label='Search Iteration', 
                                    perf_col=perf_col)
    # plt.tight_layout()
    fig.savefig(output_path / f'{tag}perf_for_top_50_every_epoch.png', dpi=450)
    plt.show()

    ## Param-size vs Performance
    fig, ax = generate_scatter_plot(ev_data_df, rs_data_df, 
                                    x_col='params', x_label='Number of Parameters (in Millions)', 
                                    perf_col=perf_col)
    # plt.tight_layout()
    plt.savefig(output_path / f'{tag}perf_vs_params_for_highperf_cand.png', dpi=450)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path',
                        type=str, nargs='?',
                        help='path of file with evolution search results')
    parser.add_argument('--input-path-random',
                        type=str, nargs='?',
                        help='path of file with random search results')
    parser.add_argument('--result_dir', type=str, nargs='?',
                        help='name of folder where plot files will be dumped')
    parser.add_argument('--tag', type=str, nargs='?',
                        help='Tag to prepend to output filename')
    args = parser.parse_args()

    plot_all(args.input_path, args.input_path_random, args.result_dir)


if __name__ == '__main__':
    main()