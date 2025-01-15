# system lib
import os.path as osp
import sys
import json

# 3rd party lib
from tqdm import tqdm
# user lib

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
print(f"Base directory: {BASE_DIR}")

train_annotations_path = osp.join("dataset", "v2_Annotations_Train_mscoco", "v2_mscoco_train2014_annotations.json")
val_annotations_path = osp.join("dataset", "v2_Annotations_Val_mscoco", "v2_mscoco_val2014_annotations.json")

train_questions_path = osp.join("dataset", "v2_Questions_Train_mscoco", "v2_OpenEnded_mscoco_train2014_questions.json")
val_questions_path = osp.join("dataset", "v2_Questions_Val_mscoco", "v2_OpenEnded_mscoco_val2014_questions.json")
test_questions_path = osp.join("dataset", "v2_Questions_Test_mscoco", "v2_OpenEnded_mscoco_test2015_questions.json")

def load_train_data():
    train_data = []
    with open(train_annotations_path, "r") as f:
        annots = json.load(f)
    train_annots = annots["annotations"]

    for annot in tqdm(train_annots, desc="Processing annotations", ncols=100, ascii=True):
        image_id = annot["image_id"]
        question_id = annot["question_id"]

        question_type = annot["question_type"]
        multiple_choice_answer = annot["multiple_choice_answer"]

        answers = []

        for answer in annot["answers"]:
            answers.append({
                "answer": answer["answer"],
                "answer_confidence": answer["answer_confidence"],
                "answer_id": answer["answer_id"]
            })

        train_data.append({
            "image_id": image_id,
            "question_id": question_id,
            "question_type": question_type,
            "multiple_choice_answer": multiple_choice_answer,
            "answers": answers
        })
    print_json(train_data[0])
    return train_data

def load_val_data():
    val_data = []
    with open(val_annotations_path, "r") as f:
        annots = json.load(f)
    val_annots = annots["annotations"]

    for annot in tqdm(val_annots, desc="Processing annotations", ncols=100, ascii=True):
        image_id = annot["image_id"]
        question_id = annot["question_id"]

        question_type = annot["question_type"]
        multiple_choice_answer = annot["multiple_choice_answer"]

        answers = []

        for answer in annot["answers"]:
            answers.append({
                "answer": answer["answer"],
                "answer_confidence": answer["answer_confidence"],
                "answer_id": answer["answer_id"]
            })

        val_data.append({
            "image_id": image_id,
            "question_id": question_id,
            "question_type": question_type,
            "multiple_choice_answer": multiple_choice_answer,
            "answers": answers
        })
    
    print_json(val_data[0])
    return val_data

def print_json(data: dict):
    print(json.dumps(data, indent=4, sort_keys=True))

def print_json_structure(json_obj, indent=0, processed=set()):
    # If the object is a dictionary, print its keys and their types
    if isinstance(json_obj, dict):
        print(' ' * indent + "{")
        for key, value in json_obj.items():
            # Print the key and its type only if it has not been processed
            if key not in processed:
                print(' ' * (indent + 2) + f'"{key}": {type(value).__name__}')
                processed.add(key)  # Mark this key as processed
            print_json_structure(value, indent + 4, processed)  # Recursively print the structure of the value
        print(' ' * indent + "}")
    
    # If the object is a list, print the structure of its elements recursively
    elif isinstance(json_obj, list):
        print(' ' * indent + "[")
        if json_obj:  # Check if the list is not empty
            print_json_structure(json_obj[0], indent + 2, processed)  # Only show structure of one element in the list
        print(' ' * indent + "]")
    
    else:
        # For basic data types, just print the type
        print(' ' * indent + f'{type(json_obj).__name__}')


if __name__ == "__main__":
    # filename = val_questions_path
    # with open(filename, 'r') as file:
    #     data = json.load(file)
    # print_json_structure(data)
    load_val_data()
    pass