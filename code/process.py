import glob
import json
import math
import traceback
import warnings

import numpy as np
import torch

from config.config_parser import *
from data.TeethInferDataset import TeethInferDataset
from models.centroids_prediction import Pointnet2MSG as CentroidNet
from models.pct_models import PctPatchRefine
from models.teeth_classification import TeethClassifier
from models.teeth_gingival_seg import Pointnet2MSG
from utils import postprocessing
from utils.cfdp import get_clustered_centroids

warnings.filterwarnings("ignore", category=UserWarning)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ScanSegmentation():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        self.model_teeth_gingival_seg = Pointnet2MSG(3)
        self.model_centroid_prediction = CentroidNet()
        self.model_refine_pct = PctPatchRefine()
        self.model_cls = TeethClassifier()

        torch.random.manual_seed(20000228)
        torch.cuda.manual_seed(20000228)

        log_dir = '../logs'

        self.model_teeth_gingival_seg.load_state_dict(
            torch.load(os.path.join(log_dir, cfg_stage1()['name'], 'best.tar'))['model'])
        self.model_centroid_prediction.load_state_dict(
            torch.load(os.path.join(log_dir, cfg_stage2()['name'], 'best.tar'))['model'])
        self.model_refine_pct.load_state_dict(
            torch.load(os.path.join(log_dir, cfg_stage3()['name'], 'best.tar'))['model'])
        self.model_cls.load_state_dict(torch.load(os.path.join(log_dir, cfg_stage4()['name'], 'best.tar'))['model'])

    @staticmethod
    def load_input(input_dir):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """

        # iterate over files in input_dir, assuming only 1 file available
        inputs = glob.glob(f'{input_dir}/*.obj')
        print("scan to process:", inputs)
        return inputs

    @staticmethod
    def write_output(labels, instances, jaw):
        """
        Write to /output/dental-labels.json your predicted labels and instances
        Check https://grand-challenge.org/components/interfaces/outputs/
        """
        pred_output = {'id_patient': "",
                       'jaw': jaw,
                       'labels': labels,
                       'instances': instances
                       }

        # just for testing
        # with open('./test/expected_output.json', 'w') as fp:
        with open('/output/dental-labels.json', 'w') as fp:
            json.dump(pred_output, fp, cls=NpEncoder)

        return

    @staticmethod
    def get_jaw(scan_path):
        try:
            # read jaw from filename
            _, jaw = os.path.basename(scan_path).split('.')[0].split('_')
        except:
            # read from first line in obj file
            try:
                with open(scan_path, 'r') as f:
                    jaw = f.readline()[2:-1]
                if jaw not in ["upper", "lower"]:
                    return None
            except Exception as e:
                print(str(e))
                print(traceback.format_exc())
                return None

        return jaw

    def predict(self, inputs):
        """
        Your algorithm goes here
        """
        self.model_teeth_gingival_seg.cuda()
        self.model_teeth_gingival_seg.eval()
        self.model_centroid_prediction.cuda()
        self.model_centroid_prediction.eval()
        self.model_refine_pct.cuda()
        self.model_refine_pct.eval()
        self.model_cls.cuda()
        self.model_cls.eval()

        batch_size = 8

        # try:
        #     assert len(inputs) == 1, f"Expected only one path in inputs, got {len(inputs)}"
        # except AssertionError as e:
        #     raise Exception(e.args)
        scan_path = inputs[0]
        print(f"loading scan : {scan_path}")
        infer_set = TeethInferDataset(scan_path)
        print('infer set inited', flush=True)
        # read input 3D scan .obj
        jaw = ''
        try:
            # you can use trimesh or other any loader we keep the same order
            jaw = self.get_jaw(scan_path)
            print("jaw processed is:", jaw)
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())

        with torch.no_grad():
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # % Stage 1: all tooth seg
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            data_tensor = infer_set.get_data_tensor()
            try:
                pred_seg = torch.argmax(self.model_teeth_gingival_seg(data_tensor[:, :, 0:6]), dim=1)
                pred_seg = pred_seg.detach().cpu().numpy()[0]

                infer_set.remove_curvatures_on_tooth(pred_seg)
            except Exception as e:
                print(str(e))
                print(traceback.format_exc())

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # % Stage 2: centroid prediction
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            try:
                data_tensor = infer_set.get_data_tensor()
                kp_reg, kp_score, seed_xyz = self.model_centroid_prediction(data_tensor, False, False)

                kp_reg = kp_reg.detach().cpu().numpy()
                kp_score = kp_score.detach().cpu().numpy()

                kp_reg = kp_reg[0, kp_score[0] < 0.2, :]

                kp_reg = get_clustered_centroids(np.asarray(kp_reg, dtype=np.float64))
            except Exception as e:
                print(str(e))
                print(traceback.format_exc())

            if len(kp_reg) == 0:
                final_labels = np.zeros((data_tensor.shape[1],))
            else:
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # % Stage 3: patches refine
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                try:
                    infer_set.make_patches_centroids(kp_reg)

                    # Stage 2: patch inference
                    patches_tensor = infer_set.get_patches_tensor()

                    all_pred_seg = np.array([])
                    for i in range(math.ceil(len(patches_tensor) / batch_size)):
                        if i * batch_size == len(patches_tensor) - 1:
                            pred_seg = self.model_refine_pct(
                                patches_tensor[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1).transpose(2,
                                                                                                              1))  # [:, 0, :]
                            pred_seg = torch.argmax(pred_seg, dim=1)
                            pred_seg = pred_seg[0:1]
                        else:
                            pred_seg = self.model_refine_pct(
                                patches_tensor[i * batch_size:(i + 1) * batch_size].transpose(2, 1))  # [:, 0, :]
                            pred_seg = torch.argmax(pred_seg, dim=1)

                        # pred_seg1 +=> pred_seg2
                        pred_seg = pred_seg.detach().cpu().numpy()
                        all_pred_seg = np.array([*all_pred_seg, *pred_seg])

                    pred_seg = all_pred_seg
                    pred_seg[pred_seg > 0.5] = 1
                    pred_seg[pred_seg < 1] = 0

                    pred_seg = postprocessing.infer_labels_denoise(infer_set.patches, pred_seg)

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # % Stage 4: patches classification
                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    patches_tensor, resamples_tensor = infer_set.get_cls_patches_tensor(all_pred_seg)

                    all_pred_cls = np.array([])
                    # all_pred_quad = np.array([])
                    for i in range(math.ceil(len(patches_tensor) / batch_size)):
                        if i * batch_size == len(patches_tensor) - 1:
                            pred_cls = self.model_cls(
                                patches_tensor[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1),
                                resamples_tensor[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1))
                            pred_cls = pred_cls[0:1]
                        else:
                            pred_cls = self.model_cls(patches_tensor[i * batch_size:(i + 1) * batch_size],
                                                      resamples_tensor[i * batch_size:(i + 1) * batch_size])
                        pred_cls = pred_cls.detach().cpu().numpy()
                        pred_cls = np.argmax(pred_cls, axis=1)

                        # 0-32 -> 0-48
                        pred_cls[pred_cls > 0] = np.ceil(pred_cls[pred_cls > 0] / 8) * 10 + (
                                pred_cls[pred_cls > 0] - 1) % 8 + 1
                        all_pred_cls = np.array([*all_pred_cls, *pred_cls])
                except Exception as e:
                    print(str(e))
                    print(traceback.format_exc())
                    final_labels = np.zeros((data_tensor.shape[1],))
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # % Stage 5: post processing
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                patches_tensor, _ = infer_set.get_cls_patches_tensor(pred_seg)
                # final_labels = postprocessing.rearrange_labels2(patches_tensor[0, :, 0:3].data.cpu().numpy(),
                #                                                        patches_tensor[:, :, 6].data.cpu().numpy(),
                #                                                        all_pred_cls)
                try:
                    final_labels = postprocessing.rearrange_labels(patches_tensor[0, :, 0:3].data.cpu().numpy(),
                                                                   patches_tensor[:, :, 6].data.cpu().numpy(),
                                                                   all_pred_cls)
                except Exception as e:
                    print(str(e))
                    print(traceback.format_exc())
                    final_labels = postprocessing.rearrange_labels_backup(patches_tensor[0, :, 0:3].data.cpu().numpy(),
                                                                          patches_tensor[:, :, 6].data.cpu().numpy(),
                                                                          all_pred_cls)

            infer_set.return_back_interpolation(final_labels)

            final_labels = infer_set.class_results

        instances = np.zeros(final_labels.shape, dtype=np.int32)
        labels_unique = np.unique(final_labels[final_labels > 0])

        for label_i in range(labels_unique.shape[0]):
            if labels_unique[label_i] > 0:
                instances[final_labels == labels_unique[label_i]] = np.squeeze(
                    np.argwhere(labels_unique == labels_unique[label_i])) + 1

        labels = np.asarray(final_labels, dtype=np.int32)
        if jaw == '' or jaw is None:
            jaw = 'lower' if np.max(final_labels) > 30 else 'upper'
        elif jaw == 'lower':
            if np.max(final_labels) < 30:
                final_labels[final_labels > 0] += 20
        elif jaw == 'upper':
            if np.max(final_labels) > 30:
                final_labels[final_labels > 0] -= 20

        return labels, instances, jaw

    def process(self):
        """
        Read input from /input, process with your algorithm and write to /output
        assumption /input contains only 1 file
        """
        input = self.load_input(input_dir='/input')
        labels, instances, jaw = self.predict(input)
        self.write_output(labels=labels, instances=instances, jaw=jaw)


if __name__ == "__main__":
    print('Main process')
    ScanSegmentation().process()
