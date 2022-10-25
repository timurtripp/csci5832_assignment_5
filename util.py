from datasets import load_dataset
import random
from collections import defaultdict
from tqdm import tqdm
from torch import LongTensor
from typing import List, Dict


def get_even_datapoints(datapoints, n):
    random.seed(42)
    dp_by_label = defaultdict(list)
    for dp in tqdm(datapoints, desc='Reading Datapoints'):
        dp_by_label[dp['label']].append(dp)

    unique_labels = [0, 1, 2]

    split = n//len(unique_labels)

    result_datapoints = []

    for label in unique_labels:
        result_datapoints.extend(random.sample(dp_by_label[label], split))

    return result_datapoints


def get_snli(train=10000, validation=1000, test=1000):
    snli = load_dataset('snli')
    train_dataset = get_even_datapoints(snli['train'], train)
    validation_dataset = get_even_datapoints(snli['validation'], validation)
    test_dataset = get_even_datapoints(snli['test'], test)

    return train_dataset, validation_dataset, test_dataset


def generate_inputs(datapoints: List[Dict]) -> (List[str], LongTensor):
    """
    TODO: Generate the input sequence for the NLI classification task and the Labels Tensor for the corresponding
    datapoints.
    Each datapoint is of the form:
        {
            'premise': str,
            'hypothesis': str,
            'label': int
        }

    :return:
    """



