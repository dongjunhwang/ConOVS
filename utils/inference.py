import datetime
import logging
import time
from collections import abc
from contextlib import ExitStack
from typing import List, Union
import torch
from torch import nn
import os
from torch.nn import functional as F
from PIL import Image
import numpy as np

from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds

from detectron2.evaluation import (
    DatasetEvaluator,
    DatasetEvaluators,
    inference_context,
)

from demo.predictor import OpenVocabVisualizer


def inference_on_dataset(
    model, data_loader,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    output_dir: str, visualize: bool,
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))
    vis_save_dir = os.path.join(output_dir, 'visual')
    os.makedirs(os.path.join(output_dir, 'visual'), exist_ok=True)

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            image = inputs[0]['image'].permute((1, 2, 0)).numpy()
            visualizer = OpenVocabVisualizer(image, model.test_metadata)
            visualizer._default_font_size = visualizer._default_font_size * 2
            if visualize:
                # save the output
                panoptic_seg, segments_info = outputs[0]["panoptic_seg"]
                panoptic_seg_resize = panoptic_seg
                if image.shape[:2] != panoptic_seg.shape[:2]:
                    panoptic_seg_resize = F.interpolate(panoptic_seg[None, None, :].float(),
                                                        size=image.shape[:2]).to(torch.int32).squeeze()
                vis_output = visualizer.draw_panoptic_seg(
                    panoptic_seg_resize.to('cpu'), segments_info
                )
                vis_output.save(os.path.join(vis_save_dir, str(inputs[0]['image_id'])+'.png'))
                # save the ground truth
                # panoptic_seg = inputs[0]["pan_seg_file_name"]
                # panoptic_seg_resize = panoptic_seg
                # if image.shape[:2] != panoptic_seg.shape[:2]:
                #     panoptic_seg_resize = F.interpolate(panoptic_seg[None, None, :].float(),
                #                                         size=image.shape[:2]).to(torch.int32).squeeze()
                # vis_output = visualizer.draw_panoptic_seg(
                #     panoptic_seg_resize.to('cpu'), segments_info
                # )
                # vis_output.save(os.path.join(vis_save_dir, str(inputs[0]['image_id']) + '_gt.png'))
                if idx > 100:
                    break
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    if visualize:
        results = None
    else:
        results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results