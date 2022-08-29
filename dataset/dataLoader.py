import scipy.io as io
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class Data(Dataset):
    def __init__(self, mode='train', dataset='toutiao_cat_data'):
        data = io.loadmat('dataset/data/processedData/{}.mat'.format(dataset))
        self.X = data['X']
        self.y = data['label']
        self.lengths = data['lengths']
        self.num_words = data['num_words'].item()
        train_X, val_X, train_y, val_y, train_length, val_length = train_test_split(self.X, self.y.squeeze(), self.lengths.squeeze(),
                                                                                    test_size=0.3, random_state=1)
        val_X, test_X, val_y, test_y, val_length, test_length = train_test_split(val_X, val_y, val_length, test_size=0.5, random_state=2)
        if mode == 'train':
            self.X = train_X
            self.y = train_y
            self.lengths = train_length
        elif mode == 'val':
            self.X = val_X
            self.y = val_y
            self.lengths = val_length
        elif mode == 'test':
            self.X = test_X
            self.y = test_y
            self.lengths = test_length

    def __getitem__(self, item):
        return self.X[item], self.y[item], self.lengths[item]

    def __len__(self):
        return self.X.shape[0]


class getDataLoader():
    def __init__(self, args):
        dataset = args.dataset
        train_data = Data('train', dataset)
        val_data = Data('val', dataset)
        test_data = Data('test', dataset)
        self.traindl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        self.valdl = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        self.testdl = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        self.num_words = train_data.num_words