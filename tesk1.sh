CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/faceseg/swin_base_ours_seg_cls_15_embed_256_grass_ce_dice.py
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/faceseg/swin_base_ours_seg_cls_15_embed_256_grass_ce_dice_focal.py
CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/faceseg/swin_base_ours_seg_cls_15_embed_256_cloud_ce_dice_focal.py