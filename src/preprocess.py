from keras.utils import image_dataset_from_directory
from config import train_directory, test_directory, image_size, batch_size, validation_split

def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    return train_dataset, validation_dataset, test_dataset

def get_transfer_datasets():
    transfer_directory = 'kaggle/Doom_vs_AnimalCrossing'
    transfer_image_size = image_size
    transfer_batch_size = batch_size
    transfer_validation_split = 0.3  # 70% train, 30% val (val also used as test)

    print('transfer train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        transfer_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=transfer_batch_size,
        image_size=transfer_image_size,
        validation_split=transfer_validation_split,
        subset='both',
        seed=47
    )

    # Reuse validation set as test set (no separate test split available)
    test_dataset = validation_dataset

    return train_dataset, validation_dataset, test_dataset