# system lib
import os
import os.path as osp
import sys
import json
import re
import time
# 3rd party lib
from tqdm import tqdm
# user lib

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

train_images_path = osp.join("dataset", "train2014")
val_images_path = osp.join("dataset", "val2014")
test_images_path = osp.join("dataset", "test2015")

train_annotations_path = osp.join("dataset", "v2_Annotations_Train_mscoco", "v2_mscoco_train2014_annotations.json")
val_annotations_path = osp.join("dataset", "v2_Annotations_Val_mscoco", "v2_mscoco_val2014_annotations.json")

train_questions_path = osp.join("dataset", "v2_Questions_Train_mscoco", "v2_OpenEnded_mscoco_train2014_questions.json")
val_questions_path = osp.join("dataset", "v2_Questions_Val_mscoco", "v2_OpenEnded_mscoco_val2014_questions.json")
test_questions_path = osp.join("dataset", "v2_Questions_Test_mscoco", "v2_OpenEnded_mscoco_test2015_questions.json")

output_folder = osp.join("dataset", "generated_yes_no")

generated_train_dataset_path = osp.join(output_folder, "train_dataset.json")
generated_val_dataset_path = osp.join(output_folder, "val_dataset.json")
generated_test_dataset_path = osp.join(output_folder, "test_dataset.json")

def print_json(data: dict):
    print(json.dumps(data, indent=4, sort_keys=True))

def print_json_structure(json_obj, indent=0, processed=set()):
    if isinstance(json_obj, dict):
        print(' ' * indent + "{")
        for key, value in json_obj.items():
            if key not in processed:
                print(' ' * (indent + 2) + f'"{key}": {type(value).__name__}')
                processed.add(key)  
            print_json_structure(value, indent + 4, processed) 
        print(' ' * indent + "}")
    elif isinstance(json_obj, list):
        print(' ' * indent + "[")
        if json_obj:  
            print_json_structure(json_obj[0], indent + 2, processed)  
        print(' ' * indent + "]")
    else:
        print(' ' * indent + f'{type(json_obj).__name__}')

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"'{func.__name__}' executed in {elapsed_time:.4f} seconds.")
        return result
    return wrapper

def generate_questions_dict(path: str):
    questions_dict = {}
    question_image_dict = {}
    with open(path, "r") as f:
        questions = json.load(f)
    for question in questions["questions"]:
        questions_dict[question["question_id"]] = question["question"]
        question_image_dict[question["question_id"]] = question["image_id"]
    
    questions_dict_path = osp.basename(osp.dirname(path)).split("_")[2].lower() + "_questions.json"
    questions_dict_path = osp.join(output_folder, questions_dict_path)
    save_dict_to_path(questions_dict, questions_dict_path)

    question_image_dict_path = osp.basename(osp.dirname(path)).split("_")[2].lower() + "_question_image.json"
    question_image_dict_path = osp.join(output_folder, question_image_dict_path)
    save_dict_to_path(question_image_dict, question_image_dict_path)

    return questions_dict_path, question_image_dict_path

def generate_images_dict(folder_path: str, output_folder: str = output_folder):
    id_to_path = {}
    
    folder_identifier = os.path.basename(folder_path)
    id_pattern = re.compile(rf"COCO_{folder_identifier}_(\d+)\.jpg")
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"): 
            match = id_pattern.match(filename)
            if match:
                image_id = int(match.group(1))
                full_path = os.path.join(folder_path, filename)
                id_to_path[image_id] = full_path
    new_path = osp.join(output_folder, f"{folder_identifier}_images.json")
    save_dict_to_path(id_to_path, new_path)
    return new_path

def save_dict_to_path(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Dictionary saved to {path}")

@timer
def generate_dataset(images_folder: str, annotations_path: str | None, questions_path: str, output_path: str):
    dataset = []
    images_dict = generate_images_dict(images_folder)
    questions_path, question_image_path = generate_questions_dict(questions_path)

    with open(questions_path, "r") as f:
        questions = json.load(f)
    with open(question_image_path, "r") as f:
        question_image = json.load(f)
    with open(images_dict, "r") as f:
        images = json.load(f)
    
    if annotations_path is not None:
        with open(annotations_path, "r") as f:
            annotations = json.load(f)["annotations"]

        for annot in tqdm(annotations, desc="Processing annotations", ncols=100, ascii=True):
            for answer in annot["answers"]:
                if answer["answer"].lower() in ["yes", "no"]: 
                    dataset.append({
                        "id": annot["question_id"],
                        "image_path": images[str(annot["image_id"])],
                        "question": questions[str(annot["question_id"])],
                        "answer": answer["answer"],
                    })
    else:
        for question_id, question_text in questions.items():
            dataset.append({
                "id": question_id,
                "image_path": images[str(question_image[question_id])],
                "question": question_text,
                "answer": "",
            })

    save_dict_to_path(dataset, output_path)


if __name__ == "__main__":
    # filename = val_questions_path
    # with open(filename, 'r') as file:
    #     data = json.load(file)
    # print_json_structure(data)

    # generate train
    generate_dataset(train_images_path, 
                     train_annotations_path, 
                     train_questions_path, 
                     generated_train_dataset_path) 
    # generate val
    generate_dataset(val_images_path, 
                     val_annotations_path, 
                     val_questions_path, 
                     generated_val_dataset_path)
    
    # generate test
    generate_dataset(test_images_path, 
                     None, 
                     test_questions_path, 
                     generated_test_dataset_path)
