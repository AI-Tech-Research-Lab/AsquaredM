from collections import defaultdict
from train_utils import get_dataset
from torch.utils.data import Dataset

def check_class_frequencies(dataset):
    """
    Check and print the frequencies of samples per class in the given dataset.

    Args:
        dataset (Dataset): The dataset to check.
    
    Returns:
        class_frequencies (dict): A dictionary with classes as keys and their frequencies as values.
    """
    class_frequencies = defaultdict(int)

    # Iterate through the dataset to count the number of samples per class
    for _, target in dataset:
        class_frequencies[target] += 1

    # Print the class frequencies
    for cls, freq in class_frequencies.items():
        print(f"Class {cls}: {freq} samples")

    return class_frequencies

train_set, val_set, test_set, input_size, classes = get_dataset(name='cifar10', model_name=None, augmentation=False, resolution=32, val_split=0.5, balanced_val=True)
print(check_class_frequencies(train_set))