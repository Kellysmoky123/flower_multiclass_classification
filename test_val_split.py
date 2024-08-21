import os
import random
import shutil
from math import ceil

def create_directory_structure(base_dir, classes):
    for class_name in classes:
        os.makedirs(os.path.join(base_dir, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'test', class_name), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'validation', class_name), exist_ok=True)

def move_files(file_list, source_dir, destination_dir, class_name):
    for file_name in file_list:
        shutil.move(os.path.join(source_dir, class_name, file_name), os.path.join(destination_dir, class_name, file_name))

def split_data(source_dir, base_dir):
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    create_directory_structure(base_dir, classes)
    
    for class_name in classes:
        all_files = [f for f in os.listdir(os.path.join(source_dir, class_name)) if os.path.isfile(os.path.join(source_dir, class_name, f))]
        total_files = len(all_files)
        
        test_count = ceil(total_files * 0.1)
        validation_count = ceil(total_files * 0.1)
        train_count = total_files - test_count - validation_count
        
        random.shuffle(all_files)
        
        test_files = all_files[:test_count]
        validation_files = all_files[test_count:test_count + validation_count]
        train_files = all_files[test_count + validation_count:]
        
        move_files(test_files, source_dir, os.path.join(base_dir, 'test'), class_name)
        move_files(validation_files, source_dir, os.path.join(base_dir, 'validation'), class_name)
        move_files(train_files, source_dir, os.path.join(base_dir, 'train'), class_name)

if __name__ == "__main__":
    source_directory = '/home/radiant/Downloads/flower_images'
    base_directory = '/home/radiant/Downloads/flower_dataset'
    split_data(source_directory, base_directory)
