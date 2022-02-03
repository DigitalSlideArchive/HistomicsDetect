from typing import List, Union

import tensorflow as tf
import numpy as np
import json

from histomics_detect.models.lnms_model import LearningNMS
from histomics_detect.anchors.create import create_anchors
from histomics_detect.metrics.iou import iou, greedy_iou_mapping
from histomics_detect.boxes.transforms import filter_edge_boxes
from histomics_detect.models.compression_network import CompressionNetwork


def save_history(history_callback, model: tf.keras.Model, experiment_id: int,
                 history_path: str = "JsonLogs/logs_{}.json", weight_path: str = "lnms_weights/cpk_{}") -> None:
    with open(history_path.format(experiment_id), 'w') as file:
        json.dump(history_callback.history, file)

    model.save_weights(weight_path.format(experiment_id))


def validate_model(ds_validation_roi: tf.data.Dataset, model: LearningNMS, faster_model, experiment_id: int = 0,
                   path: str = "JsonLogs/logs_{}_auc.json"):
    tp_counter, fp_counter, fn_counter = np.float32(0), np.float32(0), np.float32(0)
    aucs = dict()
    average_auc = np.float32(0)
    counter = np.float32(0)

    for data in ds_validation_roi:
        (rgb, boxes, name) = data
        boxes = boxes.to_tensor()
        width, height, _ = tf.shape(rgb)

        model.anchors = create_anchors(model.anchor_px, model.field, height, width)

        features, rpn_boxes, scores, nms_output = model(data)

        filtered_boxes = tf.squeeze(
            tf.gather(rpn_boxes, tf.where(tf.greater(tf.reshape(nms_output, -1), 0.5))))
        if tf.size(filtered_boxes) > 0:
            try:
                rpn_boxes_final = faster_model.align(filtered_boxes, features, faster_model.field, faster_model.pool,
                                                     faster_model.tiles)

                boxes = filter_edge_boxes(boxes, tf.shape(rgb)[1], tf.shape(rgb)[0], 32,
                                          tf.constant(False, tf.bool))

                rpn_boxes_final, condition = filter_edge_boxes(rpn_boxes_final, tf.shape(rgb)[1], tf.shape(rgb)[0], 32,
                                                               tf.constant(False, tf.bool))
                scores = tf.reshape(tf.gather(scores, tf.where(condition)), (-1, 1))

                if tf.size(rpn_boxes_final) > 0:
                    ious = iou(rpn_boxes_final, boxes)

                    tp, fp, fn, tp_list, fp_list, fn_list = greedy_iou_mapping(ious, 0.18)
                    tp, fp, fn = tp.numpy(), fp.numpy(), fn.numpy()
                else:
                    tp, fp, fn = 0, 0, tf.shape(boxes)[0].numpy()
            except:
                tp, fp, fn = 0, 0, tf.shape(boxes)[0].numpy()

            try:
                scores = tf.reshape(scores, -1)
                obj = tf.stack([1 - scores, scores], axis=1)
                # auc = greedy_pr_auc(obj, rpn_boxes_final, boxes, delta=0.1, min_iou=0.18)
                # aucs[str(name.numpy())] = auc.numpy().item()
                # average_auc += auc.numpy()
            except:
                aucs[str(name.numpy())] = 0
        else:
            tp, fp, fn = 0, 0, tf.shape(boxes)[0].numpy()
            aucs[str(name.numpy())] = 0

        tp_counter += tp
        fp_counter += fp
        fn_counter += fn
        counter += 1

    log = {
        'counter': counter,
        'fp': fp_counter.item(),
        'tp': tp_counter.item(),
        "fn": fn_counter.item(),
        # 'aucs': aucs,
        # 'auc': average_auc.item(),
        # 'avg_auc': average_auc.item() / counter
    }

    for key, value in log.items():
        print(key, value, type(value))

    for key, value in log['aucs'].items():
        print(str(key), value, type(value))

    print(log)

    with open(path.format(experiment_id), 'w') as file:
        json.dump(log, file)


def run_experiments(ds_train_roi, ds_validation_roi, callbacks: List[tf.keras.callbacks.Callback], configs: dict,
                    faster_model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
                    changing_variable: str, values: List[Union[str, tf.keras.losses.Loss, int, float, tf.Tensor]],
                    experiment_start_id: int = 0, epochs: int = 100, steps_per_epoch: int = 92) -> None:
    experiment_id = experiment_start_id
    for value in values:
        temp_configs = configs.copy()
        temp_configs[changing_variable] = value

        compression_net = CompressionNetwork(temp_configs['feature_size'], temp_configs['anchor_size'],
                                             faster_model.backbone)

        model = LearningNMS(configs, faster_model.rpnetwork, faster_model.backbone, compression_net.compression_layers,
                            [temp_configs['width'], temp_configs['height']], )
        model.compile(optimizer=optimizer)

        try:
            history_callback = model.fit(x=ds_train_roi, batch_size=1, epochs=epochs, verbose=1,
                                         callbacks=callbacks, steps_per_epoch=steps_per_epoch)  # 92
            try:
                save_history(history_callback, model, experiment_id)
                validate_model(ds_validation_roi, model, faster_model, experiment_id)
            except:
                print("validation or save failed")
        except:
            print("training failed")

        experiment_id += 1
