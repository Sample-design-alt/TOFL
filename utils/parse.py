import argparse


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--K', type=int, default=4, help='Number of augmentation for each sample')  # Bigger is better.
    parser.add_argument('--alpha', type=float, default=0.5, help='Past-future split point')

    parser.add_argument('--feature_size', type=int, default=64,
                        help='feature_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')
    parser.add_argument('--patience', type=int, default=200,
                        help='training patience')
    parser.add_argument('--aug_type', type=str, default='none', help='Augmentation type')

    parser.add_argument('--class_type', type=str, default='3C', help='Classification type')
    parser.add_argument('--gpu', type=str, default='1', help='gpu id')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    # model dataset
    parser.add_argument('--dataset_name', type=str, default='XJTU',
                        choices=['CricketX',
                                 'UWaveGestureLibraryAll',
                                 'InsectWingbeatSound',
                                 'MFPT', 'XJTU',
                                 'EpilepticSeizure',
                                 'TSC',
                                 ],
                        help='dataset')
    parser.add_argument('--nb_class', type=int, default=3,
                        help='class number')

    # ucr_path = '../datasets/UCRArchive_2018'
    parser.add_argument('--ucr_path', type=str, default='./datasets/',
                        help='Data root for dataset.')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='Data path for checkpoint.')
    # method
    parser.add_argument('--backbone_name', type=str, default='GTF', choices=['SimConv4','GTF', 'Inception', 'Mamba', 'Transformer','Resnet'])
    parser.add_argument('--val', type=bool, default=False)
    parser.add_argument('--model_name', type=str, default='SemiTime',
                        choices=['SemiSOP', 'supervised', 'TOFL', 'MTL','SemiTime'], help='choose method')

    opt = parser.parse_args()
    return opt
