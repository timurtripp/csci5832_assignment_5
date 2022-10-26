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


def generate_inputs(datapoints: List[Dict], sep_token: str = '[SEP]') -> (List[str], List[int]):
    """
    TODO: Generate the input sequence for the NLI classification task and the Labels Tensor for the corresponding
    datapoints.
    Each datapoint is of the form:
        {
            'premise': str,
            'hypothesis': str,
            'label': int
        }

    a) create the input sequence as a concatenation of: {premise, [SEP], and hypothesis}
    b) create an int list of the labels for each datapoint

    """
    raise NotImplementedError


def test_generate_input():
    dps = [
        {'premise': 'Hello', 'hypothesis': 'World', 'label': 0},
        {'premise': 'World', 'hypothesis': 'Hello', 'label': 1},
    ]

    inp_expected = (['Hello [SEP] World', 'World [SEP] Hello'], [0, 1])

    inp_sequence, labels = generate_inputs(dps, sep_token='[SEP]')

    assert inp_sequence == inp_expected[0]
    assert labels == inp_expected[1]
