from optim.pretrain import *
from dataloader.ucr2018 import load_ucr2018
import yaml
from model.inception.inceptiontime import InceptionTime
from utils.gramplot import plot
import torch
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
import matplotlib.pyplot as plt

ucr_path = './datasets/'
dataset_name = 'EpilepticSeizure'
model = 'supervised'

if model == 'TOFL':
    ckpt_backbone = r'ckpt/exp-cls/SemiSOP/EpilepticSeizure/magnitude_warp_time_warp/label1_0.3/2/backbone_best.tar'
    ckpt_cls_head = r'ckpt/exp-cls/SemiSOP/EpilepticSeizure/magnitude_warp_time_warp/label1_0.3/2/classification_head_best.tar'
elif model == 'supervised':
    ckpt_backbone = r'ckpt/exp-cls/supervised/EpilepticSeizure/magnitude_warp_time_warp/label1/2/backbone_best.tar'
    ckpt_cls_head = r'ckpt/exp-cls/supervised/EpilepticSeizure/magnitude_warp_time_warp/label1/2/classification_head_best.tar'

config_path = 'experiments/config/{0}.yaml'.format(model)
configuration = yaml.safe_load(open(config_path, 'r'))
x_train, y_train, x_val, y_val, x_test, y_test, nb_class, _ = load_ucr2018(ucr_path, dataset_name)
tensor_transform = transforms.ToTensor()
val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)
test_loader_lineval = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)


class network(torch.nn.Module):
    def __init__(self):
        super(network, self).__init__()
        if model == 'supervised':
        # backbone = InceptionTime(1, nb_class)
            backbone = SimConv4(config=configuration)
            checkpoint = torch.load(ckpt_backbone, map_location='cpu')
            backbone.load_state_dict(checkpoint)
            cls_head = torch.nn.Linear(configuration['model_params']['feature'], nb_class)
            checkpoint1 = torch.load(ckpt_cls_head, map_location='cpu')
            cls_head.load_state_dict(checkpoint1)
        elif model == 'TOFL':
            backbone = InceptionTime(1, nb_class)
            checkpoint = torch.load(ckpt_backbone, map_location='cpu')
            backbone.load_state_dict(checkpoint)
            cls_head = torch.nn.Sequential(
                torch.nn.Linear(configuration['model_params']['feature'], nb_class)
            )
            checkpoint1 = torch.load(ckpt_cls_head, map_location='cpu')
            cls_head.load_state_dict(checkpoint1)

        self.net = torch.nn.Sequential(
            backbone,
            cls_head,
        )

    def forward(self, x):
        return self.net(x)


mymo = network()

for i, (data, target) in enumerate(test_loader_lineval):
    # X_test = [0, 1]
    # data = data.reshape(-1,1)
    # data=data
    mymo.eval()
    target = target.reshape(-1, 1)
    #pred_label = mymo(data)
    pred_prob = torch.nn.functional.softmax(mymo(data))   #yuce gailv
    pred_label = torch.argmax(pred_prob).tolist()             #yuce biaoqian
    pred_prob = pred_prob[0,pred_label].tolist()
    int_mod = TSR(mymo, data.shape[-2], data.shape[-1], method='GRAD', mode='time')
    # item= np.array([test_x[0,:,:]])
    # label=int(np.argmax(test_y[0]))
    # item = np.array([test_x[0, :, :]])
    # label = int(np.argmax(test_y[0]))
    with torch.no_grad():  # 节约内存性能，with是自动处理对文件的关闭操作
        exp = int_mod.explain(data, labels=target, TSR=True)

        from scipy.interpolate import interp1d

        sample = 4000
        original_time = np.linspace(0, 1, data.shape[-2])
        new_time = np.linspace(0, 1, sample)
        data = interp1d(original_time, data.flatten(), kind='linear')(new_time)
        exp = interp1d(original_time, exp.flatten(), kind='linear')(new_time)
        cmap = plt.get_cmap('hot')
        fig = plt.figure()
        plt.scatter(range(0, sample, 1), data, c=exp, cmap=cmap)
        plt.plot(range(0, sample, 1), data)
        plt.xticks([])  # 去掉x轴
        plt.yticks([])  # 去掉y轴
        # plt.axis('off')
        plt.savefig('./img/ep/{2}/{1}_true_label{0}_pred_label{3}_pred_prob{4}.svg'.format(target.item(), i, model, pred_label, round(pred_prob, 4)), format='svg')

    # plot(np.array([data[0, :, :]]), exp, save='./img/ep/{0}.svg'.format(i))

#
# from gradcam.feature_extraction.CAMs import CAM
# from gradcam.utils.visualization import CAMFeatureMaps
#
# feature_maps = CAMFeatureMaps(CAM)
#
# extracting_module = 1
# targeting_layer = 1
# feature_maps.load(backbone.inception_blocks, backbone.inception_blocks, ['inception_blocks', 'adaptive_avg_pool'])

# checkpoint = torch.load(ckpt_cls_head, map_location='cpu')

# sup_head = torch.nn.Sequential(
#     torch.nn.Linear(configuration['model_params']['feature'], nb_class),
# ).cuda()
# checkpoint['0.weight'] = checkpoint['weight']
# checkpoint['0.bias'] = checkpoint['bias']
# del checkpoint['bias']
# del checkpoint['weight']
#
# sup_head.load_state_dict(checkpoint)
# acc_vals = list()
#
# all_sample_X = []
# all_sample_Y = []
from fastai.imports import *
# from fastai.basics import *
from fastai.callback.hook import *
# from fastai.vision.data import get_grid
# from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm
# import matplotlib.cm as cm
#
# from timeseries.all import *
#
# for i, (data, target) in enumerate(test_loader_lineval):
#     X_test = [0, 1]
#     mask = feature_maps.show(data.numpy(), None, 1, r'./gram-cam')
#     feature_maps.map_activation_to_input(mask)
#
