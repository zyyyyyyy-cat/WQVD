import torch
import torch.nn as nn
from .backbones.vit_pytorch import vit_base_in, vit_ics_lup
from .modules import A1LocalEnhancement, A2FeatureModulation
from utils.checkpoint import load_model_checkpoint

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.feature_aug_type = cfg.MODEL.FEATURE_AUG.TYPE.lower()
        if self.feature_aug_type not in ('none', 'a1', 'a2'):
            raise ValueError('FEATURE_AUG.TYPE must be none, a1, or a2')
        if pretrain_choice not in ('imagenet', 'uad_init', 'resume', 'none', 'self', 'finetune'):
            raise ValueError('Unknown PRETRAIN_CHOICE: {}'.format(pretrain_choice))
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base=factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)

        if pretrain_choice == 'imagenet' and not cfg.SOLVER.RESUME:
            self.base.load_param(model_path)
            print('Loading pretrained model......from {}'.format(model_path))

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        aug = cfg.MODEL.FEATURE_AUG
        if self.feature_aug_type == 'a1':
            self.feature_aug = A1LocalEnhancement(
                self.in_planes, aug.DIM, aug.NUM_QUERY, aug.TOPK,
                aug.NUM_HEADS, aug.LOCAL_TAU, aug.GAMMA)
            self.local_bottleneck = nn.BatchNorm1d(self.in_planes)
            self.local_bottleneck.bias.requires_grad_(False)
            self.local_bottleneck.apply(weights_init_kaiming)
            self.local_classifier = nn.Linear(self.in_planes, num_classes, bias=False)
            self.local_classifier.apply(weights_init_classifier)
        elif self.feature_aug_type == 'a2':
            self.feature_aug = A2FeatureModulation(
                self.in_planes, aug.DIM, aug.NUM_QUERY, aug.NUM_HEADS,
                aug.GAMMA, aug.MASK_OUT_INIT_STD)
        self.checkpoint_metadata = {
            'version': 1, 'type': self.feature_aug_type,
            'backbone': cfg.MODEL.TRANSFORMER_TYPE, 'in_planes': self.in_planes,
            'num_classes': num_classes,
            'backbone_config': {
                'train_size': list(cfg.INPUT.SIZE_TRAIN),
                'test_size': list(cfg.INPUT.SIZE_TEST),
                'stride_size': list(cfg.MODEL.STRIDE_SIZE),
                'sie_camera': cfg.MODEL.SIE_CAMERA, 'sie_view': cfg.MODEL.SIE_VIEW,
                'sie_coe': cfg.MODEL.SIE_COE,
            },
            'feature_aug': dict(aug) if self.feature_aug_type != 'none' else {},
        }
        if pretrain_choice in ('uad_init', 'finetune') and not cfg.SOLVER.RESUME:
            load_model_checkpoint(self, model_path, mode='uad_init')
        elif pretrain_choice == 'self' and not cfg.SOLVER.RESUME:
            load_model_checkpoint(self, model_path, mode='eval')

    def _extract_feature(self, x):
        if self.feature_aug_type == 'none':
            return self.base(x), None
        cls, patches = self.base(
            x, return_tokens='l11' if self.feature_aug_type == 'a1' else 'l12')
        if self.feature_aug_type == 'a1':
            return self.feature_aug(cls, patches)
        return self.feature_aug(cls, patches), None

    def diagnostic_features(self, x):
        """Eval-only raw features; no BN, parameter changes, or training ablation."""
        if self.training:
            raise RuntimeError('diagnostic_features requires model.eval()')
        if self.feature_aug_type == 'none':
            return {'cls': self.base(x)}
        if self.feature_aug_type != 'a1':
            raise ValueError('UAD/A1 diagnostics do not support A2')
        cls, patches = self.base(x, return_tokens='l11')
        fusion, local, details = self.feature_aug(cls, patches, return_details=True)
        return {'cls': cls, 'local': local, 'fusion': fusion, **details}
        
    def forward(self, x=None, label=None, camids=None, mode=0):
        if mode==0:
            x = torch.cat(list(x), dim=0)
            global_feat, local_feat = self._extract_feature(x)
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            local_score = None
            if self.feature_aug_type == 'a1':
                local_score = self.local_classifier(self.local_bottleneck(local_feat))
            return {'cls_score': cls_score, 'global_feat': global_feat,
                    'bn_feat': feat, 'local_feat': local_feat, 'local_score': local_score}
        else:
            global_feat, _ = self._extract_feature(x)
            feat = self.bottleneck(global_feat)
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat


    def load_param(self, trained_path):
        return load_model_checkpoint(self, trained_path, mode='eval')

    def load_param_finetune(self, model_path):
        return load_model_checkpoint(self, model_path, mode='uad_init')


__factory_T_type = {
    'vit_base_in': vit_base_in,
    'vit_ics_lup': vit_ics_lup,
}

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    print('===========building transformer===========')
    return model
