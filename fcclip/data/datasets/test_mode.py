
_TEST_MODE = {
    'openvocab_cityscapes_fine_panoptic_val': {'SEMANTIC_ON': True, 'INSTANCE_ON': True, 'PANOPTIC_ON': True},
    'openvocab_ade20k_panoptic_val': {'SEMANTIC_ON': True, 'INSTANCE_ON': True, 'PANOPTIC_ON': True},
    'openvocab_coco_2017_val_panoptic_with_sem_seg': {'SEMANTIC_ON': True, 'INSTANCE_ON': True, 'PANOPTIC_ON': True},
    'openvocab_lvis_v1_val': {'SEMANTIC_ON': False, 'INSTANCE_ON': True, 'PANOPTIC_ON': False},
    'openvocab_bdd100k_val_panoptic': {'SEMANTIC_ON': True, 'INSTANCE_ON': True, 'PANOPTIC_ON': True},
    'openvocab_mapillary_vistas_panoptic_val': {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': False},
    'openvocab_pascal20_sem_seg_val': {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': False},
    'openvocab_pascal21_sem_seg_val': {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': False},
    'openvocab_pascal_ctx459_sem_seg_val': {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': False},
    'openvocab_pascal_ctx59_sem_seg_val': {'SEMANTIC_ON': True, 'INSTANCE_ON': True, 'PANOPTIC_ON': False},
    'openvocab_ade20k_full_sem_seg_val': {'SEMANTIC_ON': True, 'INSTANCE_ON': False, 'PANOPTIC_ON': False},
}

def set_test_mode(cfg, dataset_name):
    cfg.defrost()
    for tm in ['SEMANTIC_ON', 'INSTANCE_ON', 'PANOPTIC_ON']:
        if dataset_name in _TEST_MODE:
            cfg.MODEL.MASK_FORMER.TEST[tm] = _TEST_MODE[dataset_name][tm]
    cfg.freeze()

    return cfg