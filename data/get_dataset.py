import pandas as pd
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 500), pd.set_option('display.width', 1000)
import tensorflow_datasets as tfds
import tfds_korean.korsts
from tqdm import tqdm
from enum import Enum


class DatasetName(Enum):
    KORNLI  = 'kornli'
    KORSTS  = 'korsts'

class DatasetMode(Enum):
    TRAIN   = 'train'
    DEV     = 'dev'
    TEST    = 'test'

def load_dataset(dataset_name, split_name):
    if dataset_name == DatasetName.KORNLI.value:
        dataset = tfds.load(dataset_name, split=split_name)
    elif dataset_name == DatasetName.KORSTS.value:
        dataset = tfds.load(dataset_name, split=split_name)

    return dataset

def get_dataset_from_split(dataset_name, split_name=None) -> pd.DataFrame:
    dataset = load_dataset(dataset_name, split_name)

    # label: 0 (entailment), 1 (neutral), 2 (contradiction)
    ALLOW_LABEL_NUMS = [0, 2]
    BLOCK_LABEL_NUM = 1
    sentences = {}
    for batch in tqdm(dataset.as_numpy_iterator(), desc=f"reading {split_name}"):
        sentence1: str = batch['sentence1'].decode('utf8')  # root key
        sentence2: str = batch['sentence2'].decode('utf8')  # value

        if sentence1 not in sentences:
            sentences[sentence1] = {}

        if DatasetMode.TRAIN.value in split_name:
            gold_label: int = batch['gold_label']  # sub key
            if gold_label != BLOCK_LABEL_NUM:  # not neutral
                sentences[sentence1][gold_label] = sentence2
        else:
            score: int = batch['score']  # sub key
            sentences[sentence1][score] = sentence2

    if DatasetMode.TRAIN.value in split_name:
        proc_dataset = [(key, val[0], val[2]) for key, val in sentences.items() if sorted(val.keys()) == ALLOW_LABEL_NUMS]
    else:
        proc_dataset = [(key, sub_val, sub_key) for key, val in sentences.items() for sub_key, sub_val in val.items()]

    print(f"dataset length of split {split_name}: {len(proc_dataset)}")

    return pd.DataFrame(proc_dataset)


if __name__ == '__main__':
    # Save (TRAIN: KorNLI) to csv
    mnli_train_df = get_dataset_from_split(dataset_name=DatasetName.KORNLI.value, split_name='mnli_train')
    snli_train_df = get_dataset_from_split(dataset_name=DatasetName.KORNLI.value, split_name='snli_train')

    train_dfs = [mnli_train_df, snli_train_df]
    train_df = pd.concat(train_dfs, ignore_index=True)

    train_df.to_csv('./NLU/processed/train_nli_sample.tsv', index=False, header=None, sep='\t')

    # Save (DEV & TEST: KorNLI) to csv
    xnli_dev_df = get_dataset_from_split(dataset_name=DatasetName.KORSTS.value, split_name=DatasetMode.DEV.value)
    xnli_dev_df.to_csv('./NLU/processed/valid_sts_sample.tsv', index=False, header=None, sep='\t')

    xnli_test_df = get_dataset_from_split(dataset_name=DatasetName.KORSTS.value, split_name=DatasetMode.TEST.value)
    xnli_test_df.to_csv('./NLU/processed/test_sts_sample.tsv', index=False, header=None, sep='\t')