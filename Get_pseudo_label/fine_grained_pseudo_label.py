import os
import pandas as pd
import nibabel as nib
import numpy as np
import glob

def traverse_granular(first_20_chars):
    id_path = '/data8/pengzhang/IDH/final_version.xlsx'  # 替换为你的文件路径
    df = pd.read_excel(id_path)
    matched_ids = []
    granularity_columns = ['细粒度9', '细粒度8', '细粒度7', '细粒度6', '细粒度5', 
                           '细粒度4', '细粒度3', '细粒度2', '细粒度1', '粗粒度']
    for idx, row in df.iterrows():
        for col in granularity_columns:
            if pd.notna(row[col]) and row[col] in first_20_chars and row['id'] not in matched_ids:
                matched_ids.append(row['id'])
                break  
    return matched_ids


def find_word_id(file_path):
    id_path = '/data8/pengzhang/IDH/final_version.xlsx'  # 替换为你的文件路径
    df = pd.read_excel(id_path)
    ids_gross_granularity = []
    ids_position_left = []
    ids_position_right = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        content_no_spaces = ''.join(content.split())
        first_20_chars = content_no_spaces[:20]

        left_index = first_20_chars.find('左')
        right_index = first_20_chars.find('右')

        if left_index == -1 and right_index == -1:
            ids_gross_granularity = traverse_granular(first_20_chars)
            union_ids_position = ids_gross_granularity
        
        elif left_index != -1 and right_index == -1:
            ids_gross_granularity = traverse_granular(first_20_chars)
            for idx, row in df.iterrows():
                if row['方位'] == 0:
                    ids_position_left.append(row['id'])
            union_ids_position = list(set(ids_gross_granularity) & set(ids_position_left))

        elif left_index == -1 and right_index != -1:
            ids_gross_granularity = traverse_granular(first_20_chars)
            for idx, row in df.iterrows():
                if row['方位'] == 1:
                    ids_position_right.append(row['id'])
            union_ids_position = list(set(ids_gross_granularity) & set(ids_position_right))
        
        elif left_index != -1 and right_index != -1:
            if left_index < right_index:
                part1 = first_20_chars[left_index + 1:right_index]
                part2 = first_20_chars[right_index + 1:]
                part1_id = []
                part2_id = []
                
                for idx, row in df.iterrows():
                    if row['方位'] == 0:
                        ids_position_left.append(row['id'])
                for idx, row in df.iterrows():
                    if row['方位'] == 1:
                        ids_position_right.append(row['id'])
                part1_id = traverse_granular(part1)
                part2_id = traverse_granular(part2)
                union_ids_position_part1 = list(set(ids_position_left) & set(part1_id))
                union_ids_position_part2 = list(set(ids_position_right) & set(part2_id))
                union_ids_position = list(set(union_ids_position_part1) | set(union_ids_position_part2))

            else:
                part1 = first_20_chars[right_index + 1:left_index]
                part2 = first_20_chars[left_index + 1:]
                part1_id = []
                part2_id = []

                for idx, row in df.iterrows():
                    if row['方位'] == 0:
                        ids_position_left.append(row['id'])
                for idx, row in df.iterrows():
                    if row['方位'] == 1:
                        ids_position_right.append(row['id'])
                part1_id = traverse_granular(part1)
                part2_id = traverse_granular(part2)
                union_ids_position_part1 = list(set(ids_position_right) & set(part1_id))
                union_ids_position_part2 = list(set(ids_position_left) & set(part2_id))
                union_ids_position = list(set(union_ids_position_part1) | set(union_ids_position_part2))

    return union_ids_position         


def process_mri_image(input_file, output_file, target_values):
    img = nib.load(input_file)
    data = img.get_fdata()
    new_data = np.zeros(data.shape)
    for value in target_values:
        new_data[data == value] = 1
    new_img = nib.Nifti1Image(new_data, img.affine, img.header)
    nib.save(new_img, output_file)


IDH_txt_dir = '/data7/pengzhang/GBM_META_PCN_NoSkull/TXT'
IDH_txt_pathList = glob.glob(os.path.join(IDH_txt_dir, '*.txt'))

for IDH_txt_path in IDH_txt_pathList:
    IDH_txt_base_name = os.path.basename(IDH_txt_path)
    name_without_extension = os.path.splitext(IDH_txt_base_name)[0]
    MNI152_path = '/data7/pengzhang/GBM_META_PCN_NoSkull/MNI152_Hammer_atlas.nii.gz'
    target_values = find_word_id(IDH_txt_path)

    if len(target_values) == 0:
        print(IDH_txt_base_name)

    Stand_GT_Mask_name = f"{name_without_extension}_Low_MNI152_GT.nii.gz"
    output_path = os.path.join('/data7/pengzhang/GBM_META_PCN_NoSkull', Stand_GT_Mask_name)
    process_mri_image(MNI152_path, output_path, target_values)

    print(Stand_GT_Mask_name)

