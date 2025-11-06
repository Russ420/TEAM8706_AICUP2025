from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

if __name__ == '__main__':
    """
    This file reference Dataset023 to generate nnUNet-compatible file structure
    """
    base = '/work/u4526725/Dataset/aicup2025_seg'
    target_dataset_id = 306
    target_dataset_name = f'Dataset{target_dataset_id:03.0f}_aicup2025'
    maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))

    imagesTr = join(base, 'imagesTr')
    imagesTs = join(base, 'imagesTs')
    labelsTr = join(base, 'labelsTr')

    target_imagesTr = join(nnUNet_raw, target_dataset_name, 'imagesTr')
    target_imagesTs = join(nnUNet_raw, target_dataset_name, 'imagesTs')
    target_labelsTr = join(nnUNet_raw, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    # train-label
    filenames_train = subfiles(imagesTr, join=False, suffix='.nii.gz')
    filenames_label = subfiles(labelsTr, join=False, suffix='.nii.gz')
    for filename_train, filename_label in zip(filenames_train, filenames_label):
        casename_train = os.path.basename(filename_train).split('.')[0]
        casename_label = os.path.basename(filename_label).split('_')[0]
        shutil.copy(join(imagesTr, filename_train), join(target_imagesTr, casename_train + '_0000.nii.gz'))
        shutil.copy(join(labelsTr, filename_label), join(target_labelsTr, casename_label + '.nii.gz'))

    # test
    filenames_test = subfiles(imagesTs, join=False, suffix=".nii.gz")
    for filename_test in filenames_test:
        casename_test = os.path.basename(filename_test).split('.')[0]
        shutil.copy(join(imagesTs, filename_test), join(target_imagesTs, casename_test + '_0000.nii.gz'))

    class_map = {1: 'cardiac_muscle', 2: 'aortic_valve', 3: 'coronary_artery_calcification'}
    labels = {
        j: i for i, j in class_map.items()
    }
    labels['background'] = 0

    generate_dataset_json(
        join(nnUNet_raw, target_dataset_name),
        {0: 'CT'},
        labels,
        len(filenames_train),
        '.nii.gz',
        None,
        None,
        target_dataset_name,
        overwrite_image_reader_writer='NibabelIOWithReorient',
        reference='https://tbrain.trendmicro.com.tw/Competitions/Details/41',
        license='Private'
    )