from dataset.DataPreprocessing import getFormatData
import argparse
from Trainer import Trainer

parse = argparse.ArgumentParser()
parse.add_argument('--preprocess', action='store_true', help="是否预处理数据")
parse.add_argument('--dataset', type=str, help="数据集", default='toutiao_cat_data')
parse.add_argument('--batch_size', type=int, help="batch大小", default=1000)
parse.add_argument('--epochs', type=int, help="epoch大小", default=100)
parse.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
parse.add_argument('--model', type=str, choices=['lstm', 'rnn', 'gru'], default='lstm')
args = parse.parse_args()

if __name__ == "__main__":
    if args.preprocess:
        getFormatData(args.dataset)

    trainer = Trainer(args)
    if args.mode == 'train':
        trainer.train()
        trainer.test()
    elif args.mode == 'test':
        trainer.test()
