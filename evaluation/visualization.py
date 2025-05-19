from optim.pretrain import *
from dataloader.ucr2018 import load_ucr2018
import yaml
from utils.tsne import gather_all_by_tsne
from sklearn.preprocessing import MinMaxScaler
from utils.parse import parse_option
from utils.utils import log_file, Config
from utils.trainer_strategies import get_backbone

#  2022-03-17 14:23:51,238] Trial 47 finished with value: 71.79487609863281 and parameters: {'K': 8, 'alpha': 0.1, 'lr': 0.00934719358698068, 'weight_decay': 1.0765539027448453e-07, 'feature': 128, 'kernel': 7}. Best is trial 47 with value: 71.79487609863281.


ucr_path = '../datasets/'
dataset_name = 'MFPT'
model = 'TOFL'
label_ratio = 1
opt = parse_option()
config_path = '../experiments/config/{0}.yaml'.format(opt.model_name)
backbone_path = ('../experiments/config/backbones/{0}.yaml'.format(opt.backbone_name))
configuration = Config(
    backbone_file_path=backbone_path,
    default_config_file_path=config_path
).parse()

ckpt_backbone =r'/data/chenrj/paper4/ckpt/exp-cls/TOFL/{0}/magnitude_warp_time_warp/label{1}/2_vision/backbone_best.tar'.format(opt.dataset_name, label_ratio)
ckpt_cls_head =r'/data/chenrj/paper4/ckpt/exp-cls/TOFL/{0}/magnitude_warp_time_warp/label{1}/2_vision/classification_head_best.tar'.format(opt.dataset_name, label_ratio)

x_train, y_train, x_val, y_val, x_test, y_test, configuration['data_params']['nb_class'], _ = load_ucr2018(ucr_path, dataset_name,configuration)


tensor_transform = transforms.ToTensor()
val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)
test_loader_lineval = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)


backbone = get_backbone(opt, configuration)

checkpoint = torch.load(ckpt_backbone, map_location='cpu')
backbone.load_state_dict(checkpoint)

checkpoint = torch.load(ckpt_cls_head, map_location='cpu')
sup_head = torch.nn.Sequential(
    torch.nn.Linear(configuration['model_params']['d_model'], configuration['data_params']['nb_class']),
).cuda()
# checkpoint['0.weight'] = checkpoint['weight']
# checkpoint['0.bias'] = checkpoint['bias']
# del checkpoint['bias']
# del checkpoint['weight']

sup_head.load_state_dict(checkpoint)
acc_vals = list()

all_sample_X = []
all_sample_Y = []
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

print('Acc:', sum(acc_vals) / len(acc_vals))
# estimate the accuracy

all_sample_X = torch.cat(all_sample_X, dim=0).cpu().detach().numpy()
all_sample_Y = torch.cat(all_sample_Y, dim=0).cpu().detach().numpy()
print('正在绘制gather图...')
file_name = './'
gather_all_by_tsne(all_sample_X, all_sample_Y, configuration['data_params']['nb_class'], file_name + 'gather_figure')
print('gather图绘制完成！')
