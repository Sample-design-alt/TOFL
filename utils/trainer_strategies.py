from model.backbones.model_backbone import SimConv4
from model.backbones.inception.inceptiontime import InceptionTime
from model.backbones.GTF.GTF import GTF
from model.backbones.Transformer import transformer
from model.backbones.mamba import mamba
from model.backbones.Resnet import ResNet


def get_backbone(opt, cfg):
    if opt.backbone_name == 'SimConv4':
        backbone_lineval = SimConv4(config=cfg).cuda()
    elif opt.backbone_name == 'GTF':
        backbone_lineval = GTF(1, cfg['data_params']['nb_class'], configs=cfg, opt=opt).cuda()
    elif opt.backbone_name == 'Inception':
        backbone_lineval = InceptionTime(1, cfg['data_params']['nb_class'], configs=cfg, opt=opt).cuda()
    elif opt.backbone_name == 'Transformer':
        backbone_lineval = transformer(cfg).cuda()
    elif opt.backbone_name == 'Mamba':
        backbone_lineval = mamba(cfg).cuda()
    elif opt.backbone_name == 'Resnet':
        backbone_lineval = ResNet(cfg).cuda()
    else:
        raise ValueError("Expected the backbone name to be in the form 'SimConv4' or 'Inception' or 'Mamba'.")
    return backbone_lineval
