import yaml
import os

file = open(os.path.join(os.path.dirname(__file__), 'config.yml'), 'r', encoding='utf-8')
data = yaml.full_load(file.read())


def cfg_log_dir():
    return data['common']['log-dir']


def cfg_data_dir():
    return data['common']['data-dir']


def cfg_label_dir():
    return data['common']['label-dir']


def cfg_dental_model_size():
    return int(data['common']['dental-model-size'])


def cfg_patch_size():
    return int(data['common']['patch-size'])


def cfg_class_dental_model_size():
    return int(data['common']['class-dental-model-size'])


def cfg_class_patch_size():
    return int(data['common']['class-patch-size'])


def cfg_stage1():
    return data['stage1']


def cfg_stage2():
    return data['stage2']


def cfg_stage3():
    return data['stage3']


def cfg_stage4():
    return data['stage4']


if __name__ == '__main__':
    print('Log dir:', cfg_log_dir())
    print('Dental model size:', cfg_dental_model_size())
    print('Patch size:', cfg_patch_size())
    print('Stage 1:', cfg_stage1())
    print('Stage 2:', cfg_stage2())
