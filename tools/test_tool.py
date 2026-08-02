import logging
import os
from timeit import default_timer as timer

from tools.eval_tool import gen_time_str, output_value
from utils.device import get_device, move_batch_to_device, prepare_data_loader

logger = logging.getLogger(__name__)


def test(parameters, config, gpu_list):
    model = parameters["model"]
    dataset = parameters["test_dataset"]
    model.eval()
    device = get_device()
    dataset = prepare_data_loader(dataset, device)

    acc_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = "testing"

    output_time = config.getint("output", "output_time")
    step = -1
    result = []

    for step, data in enumerate(dataset):
        data = move_batch_to_device(data, device)

        results = model(data, config, gpu_list, acc_result, "test")
        result = result + results["output"]
        cnt += 1

        if step % output_time == 0:
            delta_t = timer() - start_time

            output_value(0, "test", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    delta_t = timer() - start_time
    output_info = "testing"
    output_value(0, "test", "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                 "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

    return result
