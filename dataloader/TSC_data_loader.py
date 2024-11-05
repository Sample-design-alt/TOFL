import numpy as np
from sklearn import preprocessing


def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


def TSC_data_loader(dataset_path, dataset_name):
    print("[INFO] {}".format(dataset_name))

    Train_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv')
    Test_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv')
    Train_dataset = Train_dataset.astype(np.float32)
    Test_dataset = Test_dataset.astype(np.float32)

    X_train = Train_dataset[:, 1:]
    y_train = Train_dataset[:, 0:1]

    X_test = Test_dataset[:, 1:]
    y_test = Test_dataset[:, 0:1]
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y_train, axis=1))
    y_train = le.transform(np.squeeze(y_train, axis=1))
    y_test = le.transform(np.squeeze(y_test, axis=1))
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test

# def uea_data_loader(dataset_path,dataset_name):
#     print("[INFO] {}".format(dataset_name))
#
#
#
#     Train_dataset = \
#         loadarff(open(f'{dataset_path}/{dataset_name}/{dataset_name}_TRAIN.arff', 'r', encoding='UTF-8'))[0]
#     Test_dataset = \
#         loadarff(open(f'{dataset_path}/{dataset_name}/{dataset_name}_TEST.arff', 'r', encoding='UTF-8'))[0]
#     Train_dataset = Train_dataset.astype(np.float32)
#     Test_dataset = Test_dataset.astype(np.float32)
#
#     X_train = Train_dataset[:, 1:]
#     y_train = Train_dataset[:, 0:1]
#
#     X_test = Test_dataset[:, 1:]
#     y_test = Test_dataset[:, 0:1]
#     le = preprocessing.LabelEncoder()
#     le.fit(np.squeeze(y_train, axis=1))
#     y_train = le.transform(np.squeeze(y_train, axis=1))
#     y_test = le.transform(np.squeeze(y_test, axis=1))
#
#         train_x, train_y = extract_data(train_data)
#         test_x, test_y = extract_data(test_data)
#         train_x[np.isnan(train_x)] = 0
#         test_x[np.isnan(test_x)] = 0
#
#         scaler = StandardScaler()
#         scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
#         train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
#         test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)
#
#         # 放到0-Numclass
#         labels = np.unique(train_y)
#         num_class = len(labels)
#         # print(num_class)
#         transform = {k: i for i, k in enumerate(labels)}
#         train_y = np.vectorize(transform.get)(train_y)
#         test_y = np.vectorize(transform.get)(test_y)
#
#         torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)
#
#     TrainDataset = DealDataset(train_x, train_y)
#     TestDataset = DealDataset(test_x, test_y)
#     # return TrainDataset,TestDataset,len(labels)
#     train_loader = DataLoader(dataset=TrainDataset,
#                               batch_size=args.batch_size,
#                               shuffle=True)
#     test_loader = DataLoader(dataset=TestDataset,
#                              batch_size=args.batch_size,
#                              shuffle=True)
#
#     return train_loader, test_loader, num_class
