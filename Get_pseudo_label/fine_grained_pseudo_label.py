import os
import pandas as pd
import nibabel as nib
import numpy as np
import glob

def traverse_granular(first_20_chars):
    id_path = 'final_version.xlsx' 
    df = pd.read_excel(id_path)
    matched_ids = []
    granularity_columns = ['Fine Level 9', 'Fine Level 8', 'Fine Level 7', 'Fine Level 6', 'Fine Level 5', 
                           'Fine Level 4', 'Fine Level 3', 'Fine Level 2', 'Fine Level 1', 'Coarse Level']
    for idx, row in df.iterrows():
        for col in granularity_columns:
            if pd.notna(row[col]) and row[col] in first_20_chars and row['id'] not in matched_ids:
                matched_ids.append(row['id'])
                break  
    return matched_ids


def find_word_id(file_path):
    id_path = 'final_version.xlsx' 
    df = pd.read_excel(id_path)
    ids_gross_granularity = []
    ids_position_left = []
    ids_position_right = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        content_no_spaces = ''.join(content.split())
        first_20_chars = content_no_spaces[:20]

        left_index = first_20_chars.find('right')
        right_index = first_20_chars.find('left')

        if left_index == -1 and right_index == -1:
            ids_gross_granularity = traverse_granular(first_20_chars)
            union_ids_position = ids_gross_granularity
        
        elif left_index != -1 and right_index == -1:
            ids_gross_granularity = traverse_granular(first_20_chars)
            for idx, row in df.iterrows():
                if row['Orientation'] == 0:
                    ids_position_left.append(row['id'])
            union_ids_position = list(set(ids_gross_granularity) & set(ids_position_left))

        elif left_index == -1 and right_index != -1:
            ids_gross_granularity = traverse_granular(first_20_chars)
            for idx, row in df.iterrows():
                if row['Orientation'] == 1:
                    ids_position_right.append(row['id'])
            union_ids_position = list(set(ids_gross_granularity) & set(ids_position_right))
        
        elif left_index != -1 and right_index != -1:
            if left_index < right_index:
                part1 = first_20_chars[left_index + 1:right_index]
                part2 = first_20_chars[right_index + 1:]
                part1_id = []
                part2_id = []
                
                for idx, row in df.iterrows():
                    if row['Orientation'] == 0:
                        ids_position_left.append(row['id'])
                for idx, row in df.iterrows():
                    if row['Orientation'] == 1:
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
                    if row['Orientation'] == 0:
                        ids_position_left.append(row['id'])
                for idx, row in df.iterrows():
                    if row['Orientation'] == 1:
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


txt_dir = 'Dataset/In_house_data/TXT'
txt_pathList = glob.glob(os.path.join(txt_dir, '*.txt'))

for txt_path in txt_pathList:
    txt_base_name = os.path.basename(txt_path)
    name_without_extension = os.path.splitext(txt_base_name)[0]
    MNI152_path = 'MNI152_Hammer_atlas.nii.gz'
    target_values = find_word_id(txt_path)

    if len(target_values) == 0:
        print(txt_base_name)

    Stand_GT_Mask_name = f"{name_without_extension}_MNI152_GT.nii.gz"
    output_path = os.path.join('Dataset/In_house_data/seg', Stand_GT_Mask_name)
    process_mri_image(MNI152_path, output_path, target_values)

    print(Stand_GT_Mask_name)

