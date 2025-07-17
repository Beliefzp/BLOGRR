import os
import pandas as pd
import nibabel as nib
import numpy as np
import glob


def find_word_id(file_path):
    id_path = '/data7/pengzhang/IDH/ZP_version.xlsx'  # 替换为你的文件路径
    df = pd.read_excel(id_path)
    ids_gross_granularity = []
    ids_position_left = []
    ids_position_right = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        content_no_spaces = ''.join(content.split())
        first_20_chars = content_no_spaces[:20]

        for idx, row in df.iterrows():
            if row['粗粒度'] in first_20_chars:
                ids_gross_granularity.append(row['id'])

        if '左' in first_20_chars:
            for idx, row in df.iterrows():
                if row['方位'] == 0:
                    ids_position_left.append(row['id'])

        if '右' in first_20_chars:
            for idx, row in df.iterrows():
                if row['方位'] == 1:
                    ids_position_right.append(row['id'])

    common_ids_left = list(set(ids_gross_granularity) & set(ids_position_left))

    common_ids_right = list(set(ids_gross_granularity) & set(ids_position_right))

    union_ids_position = list(set(common_ids_left) | set(common_ids_right))

    print("ID输出：", union_ids_position)
    return union_ids_position


def process_mri_image(input_file, output_file, target_values):
    img = nib.load(input_file)
    data = img.get_fdata()

    new_data = np.zeros(data.shape)

    for value in target_values:
        new_data[data == value] = 1

    new_img = nib.Nifti1Image(new_data, img.affine, img.header)

    nib.save(new_img, output_file)


IDHStd_Reg_Std_dir = '/data7/pengzhang/IDH/IDHStd_Reg_Std'
IDHStd_Reg_Std_pathList = glob.glob(os.path.join(IDHStd_Reg_Std_dir, '*.nii.gz'))
for IDHStd_Reg_Std_path in IDHStd_Reg_Std_pathList:
    ########### 先找出文本对应的id ###########
    IDHStd_Reg_Std_base_name = os.path.basename(IDHStd_Reg_Std_path)
    name_without_extension = os.path.splitext(IDHStd_Reg_Std_base_name)[0]
    name_without_extension = os.path.splitext(name_without_extension)[0]
    desired_name = name_without_extension.replace('_stand', '')
    word_name = f"{desired_name}.txt"
    word_path = os.path.join('/data7/pengzhang/IDH/txt', word_name)
    target_values = find_word_id(word_path)

    ########### 生成文本信息对应的GT ###########
    output_path = os.path.join('/data7/pengzhang/IDH/IDHStdRegStd_GTMask', os.path.basename(IDHStd_Reg_Std_path))
    process_mri_image(IDHStd_Reg_Std_path, output_path, target_values)

