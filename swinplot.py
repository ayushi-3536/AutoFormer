import argparse
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        default='C:\\Users\\ayush\\projects\\Autoformers\\AutoFormer\\swinlog.txt',
                        type=str, nargs='?',
                        help='path of file')
    parser.add_argument('--result_dir', default='C:\\Users\\ayush\\projects\\Autoformers\\AutoFormer\\', type=str, nargs='?',
                        help='name of folder where plot files will be dumped')
    args = parser.parse_args()
    # train_loss = []
    # test_psnr = []
    # for line in open(args.input_path, 'r'):
    #     lines = [i for i in line.split()]
    #     print(lines)
    #     train_loss.append(float(lines[3].replace(',', '')))
    #     test_psnr.append(float(lines[5].replace(',', '')))
    #
    #
    # print("train_loss", train_loss)
    # print("cum sum tl", train_loss)
    # print("test acc", test_psnr)
    #
    # test_acc_max = [max(test_psnr[:i+1]) for i in range(len(test_psnr))]
    # print("test acc max", test_acc_max)
    #
    # train_loss_min = [min(train_loss[:i+1]) for i in range(len(train_loss))]
    # print("train loss min", train_loss_min)
    #
    # l = [i for i in range(len(train_loss))]
    # plt.title("train loss vs epochs")
    # plt.xlabel('Epochs')
    # plt.ylabel('Train Loss')
    # plt.plot(l, train_loss_min)
    #
    # plt.tight_layout()
    # plt.savefig(f'train_loss_vs_epoch_milestone2.pdf', dpi=450)
    # plt.show()
    #
    # plt.title("test PSNR vs epochs")
    # plt.xlabel('Epochs')
    # plt.ylabel('Test PSNR')
    # plt.plot(l, test_acc_max)
    #
    # plt.tight_layout()
    # plt.savefig(f'test_acc1_vs_epoch_milestone2.pdf', dpi=450)
    # plt.show()

    data_df = pd.read_json('C:\\Users\\ayush\\projects\\Autoformers\\AutoFormer\\swinir_search.json', lines=True)
    print(list(data_df.columns.values))

    epochs = data_df['epoch']
    num_params = data_df['params']
    test_acc1 = data_df['psnr']

    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    print(epochs)
    print(test_acc1)
    plt.scatter(epochs+1, test_acc1, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'psnr_for_top_50_every_epoch_milestone2.pdf', dpi=450)
    plt.show()

    plt.xlabel('Params')
    plt.ylabel('PSNR')
    print(num_params)
    print(test_acc1)
    plt.scatter(num_params, test_acc1, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'psnr_vs_params_for_highperf_cand_milestone2.pdf', dpi=450)
    plt.show()
