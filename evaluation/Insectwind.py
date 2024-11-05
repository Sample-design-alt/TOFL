from optim.pretrain import *
from dataloader.ucr2018 import load_ucr2018
import yaml
from utils.tsne import gather_all_by_tsne
from scipy.signal import savgol_filter

#  2022-03-17 14:23:51,238] Trial 47 finished with value: 71.79487609863281 and parameters: {'K': 8, 'alpha': 0.1, 'lr': 0.00934719358698068, 'weight_decay': 1.0765539027448453e-07, 'feature': 128, 'kernel': 7}. Best is trial 47 with value: 71.79487609863281.


ucr_path = '../datasets/'
dataset_name = 'EpilepticSeizure'
ckpt_backbone = r'/data/chenrj/semi-order-time/ckpt/exp-cls/SemiSOP/EpilepticSeizure/magnitude_warp_time_warp/label1_0.6/0/backbone_best.tar'
ckpt_cls_head = r'/data/chenrj/semi-order-time/ckpt/exp-cls/SemiSOP/EpilepticSeizure/magnitude_warp_time_warp/label1_0.6/0/classification_head_best.tar'

config_path = '../experiments/config/{0}.yaml'.format('SemiSOP')
configuration = yaml.safe_load(open(config_path, 'r'))
x_train, y_train, x_val, y_val, x_test, y_test, nb_class, _ = load_ucr2018(ucr_path, dataset_name)
tensor_transform = transforms.ToTensor()
val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)
test_loader_lineval = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

backbone = Resv4(config=configuration).cuda()

checkpoint = torch.load(ckpt_backbone, map_location='cpu')
backbone.load_state_dict(checkpoint)

checkpoint = torch.load(ckpt_cls_head, map_location='cpu')
sup_head = torch.nn.Sequential(
    torch.nn.Linear(configuration['model_params']['feature'], nb_class),
).cuda()
sup_head.load_state_dict(checkpoint)
acc_vals = list()

all_sample_X = []
all_sample_Y = []
plot=False
for i, (data, target) in enumerate(test_loader_lineval):
    data = data.cuda()
    target = target.cuda()
    feature = backbone(data).detach()
    output = sup_head(feature)
    prediction = output.argmax(-1)
    correct = prediction.eq(target.view_as(prediction)).sum()
    accuracy = (100.0 * correct / len(target))
    acc_vals.append(accuracy.item())
    all_sample_X.append(feature)
    all_sample_Y.append(target)

print(sum(acc_vals) / len(acc_vals))
# estimate the accuracy

all_sample_X = torch.cat(all_sample_X, dim=0).cpu().detach().numpy()
all_sample_Y = torch.cat(all_sample_Y, dim=0).cpu().detach().numpy()
print('正在绘制gather图...')
file_name = './'
gather_all_by_tsne(all_sample_X, all_sample_Y, nb_class, file_name + 'gather_figure')
print('gather图绘制完成！')
