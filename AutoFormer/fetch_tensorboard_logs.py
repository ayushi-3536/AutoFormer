from datetime import datetime

import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import argparse
print(tf.__version__)
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Evaluation:

    def __init__(self, store_dir, name, stats=[]):
        """
        Creates placeholders for the statistics listed in stats to generate tensorboard summaries.
        e.g. stats = ["loss"]
        """
        self.folder_id = "%s-%s" % (name, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.summary_writer = SummaryWriter(os.path.join(store_dir, self.folder_id))
        self.stats = stats

    def write_episode_data(self, episode, eval_dict):
        """
         Write episode statistics in eval_dict to tensorboard, make sure that the entries in eval_dict are specified in stats.
         e.g. eval_dict = {"loss" : 1e-4}
        """

        for k in eval_dict:
            assert (k in self.stats)
            self.summary_writer.add_scalar(k, eval_dict[k], global_step=episode)

        self.summary_writer.flush()

    def close_session(self):
        self.summary_writer.close()


def read_logs(file):
    metric = ['loss', 'ppl']
    plot_data = {'loss': [], 'ppl': []}

    logs = tf.train.summary_iterator(file)
    print(logs)
    for idx, log in enumerate(logs):
        values = log.summary.value
        # print("values",values)
        # print("values:",idx,"val",valuesp[0].metric)
        if idx == 0 or idx == 1:
            continue
        if values[0].tag in metric:
            # print("values", values, "val tag", values[0].tag)
            plot_data[values[0].tag].append(values[0].simple_value)
    print(plot_data)
    plt.plot(plot_data['loss'], label="loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    plt.tight_layout()
    plt.savefig(f'valid_loss_vs_epoch_for_milestone3.pdf', dpi=450)
    plt.plot(plot_data['ppl'], label="ppl")
    plt.xlabel("epoch")
    plt.ylabel("ppl")
    plt.legend()
    plt.show()
    plt.savefig(f'valid_ppl_vs_epoch_for_milestone3.pdf', dpi=450)
    return plot_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        default='C:\\Users\\ayush\\projects\\Autoformers\\AutoFormer\\log.txt',
                        type=str, nargs='?',
                        help='path of file')
    args = parser.parse_args()
    read_logs(args.input_path)
