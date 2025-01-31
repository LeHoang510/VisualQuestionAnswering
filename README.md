# VisualQuestionAnswering
Analyse image and answer the given question by using Computer Vision and Natural Language Processing

- TODO: 
+ write readme about the project
+ train llava model (In the future)

# Project Description
This project focuses on building Visual Question Answering (VQA) models capable of answering yes/no questions based on input images. 
The models are trained using the VQA COCO dataset

To ensure a comprehensive evaluation, three different models are implemented and their results are compared. 
This approach helps analyze their effectiveness in understanding images and accurately answering binary questions, as well as evaluating the efficiency of different approaches.

# Model Details

This project is built upon the traditional VQA model (as shown in Figure 3), which consists of three main components:

- **Feature Extraction**: This step involves extracting feature vectors to represent both the textual data (question) and the visual data (image) from the input.

- **Feature Fusion**: The extracted features from both the image and the question are then combined into a single vector, which serves as a unified representation of the input.

- **Answer Classifier**: Based on the fused feature vector, the classifier predicts the true or false answer to the given question.

![Figure1](docs/infor/The-Overall-VQA-Structure.png)

### Basic Model
- CNN + LSTM 
### Advance Model
- ViT + RoBERTa
### Visual Large Language Model
- LLava 7B

# Set Up

- Link to the dataset: https://visualqa.org/
- Download the zip file from the website and extract to the folder dataset ([Dataset Structure](#dataset-structure))

- How to run
    + Run the preprocess_yes_no_dataset.py to get new json files for train/val/test set

        ```python
        python data_preprocess/preprocess_yes_no_dataset.py
        ```
    
    + Train basic model

        ```python
        python train/train_basic.py
        ```
    
    + Train advance model

        ```python
        python train/train_advance.py
        ```
    
    + Evaluate visual large language model

        ```python
        python train/train_vllm.py
        ```

# Folder Structure
```
.
├── app.py
├── data_preprocess
│   ├── generate_dataset.py
│   └── generate_yes_no_dataset.py
├── dataset
├── docs
│   ├── advance_model
│   ├── basic_model
│   ├── results
│   └── vllm_model
├── model
│   ├── utils.py
│   ├── vqa_dataset.py
│   ├── VQAModelAdvance.py
│   ├── VQAModelBasic.py
│   └── VQAVLLM.py
├── output
│   ├── basic_logs
│   ├── advance_logs
│   ├── basic_model.pth
│   ├── advance_model.pth
│   └── vllm_checkpoint.json
├── README.md
├── requirements.txt
├── test.py
├── train
│   ├── train_advance.py
│   ├── train_basic.py
│   ├── train_vllm_checkpoint.py
│   └── train_vllm.py
├── utils
│   ├── config.yaml
│   ├── huggingface.py
│   └── utils.py
├── visualize
│   ├── compare_models.py
│   └── sample_data.py
└── VQA_project.pdf
```

# Dataset Structure
```
.
└──  dataset
    ├── generated
    │   ├── test2015_images.json
    │   ├── test_dataset.json
    │   ├── test_question_image.json
    │   ├── test_questions.json
    │   ├── train2014_images.json
    │   ├── train_dataset.json
    │   ├── train_question_image.json
    │   ├── train_questions.json
    │   ├── val2014_images.json
    │   ├── val_dataset.json
    │   ├── val_question_image.json
    │   └── val_questions.json
    ├── generated_yes_no
    │   ├── test2015_images.json
    │   ├── test_dataset.json
    │   ├── test_question_image.json
    │   ├── test_questions.json
    │   ├── train2014_images.json
    │   ├── train_dataset.json
    │   ├── train_question_image.json
    │   ├── train_questions.json
    │   ├── val2014_images.json
    │   ├── val_dataset.json
    │   ├── val_question_image.json
    │   └── val_questions.json
    ├── test2015
    ├── test2015.zip
    ├── train2014
    ├── train2014.zip
    ├── v2_Annotations_Train_mscoco
    │   ├── struct.txt
    │   └── v2_mscoco_train2014_annotations.json
    ├── v2_Annotations_Val_mscoco
    │   ├── struct.txt
    │   └── v2_mscoco_val2014_annotations.json
    ├── v2_Questions_Test_mscoco
    │   ├── v2_OpenEnded_mscoco_test2015_questions.json
    │   └── v2_OpenEnded_mscoco_test-dev2015_questions.json
    ├── v2_Questions_Train_mscoco
    │   ├── struct.txt
    │   └── v2_OpenEnded_mscoco_train2014_questions.json
    ├── v2_Questions_Val_mscoco
    │   └── v2_OpenEnded_mscoco_val2014_questions.json
    ├── val2014
    └── val2014.zip
```

# Training results
- Basic model - Accuracy: 0.56
- Advance model - Accuracy: 0.53
- Visual Language model - Accuracy: 0.49

| **Model**   | **Basic** | **Advance** | **Visual Language** |
|-------------|-----------|-------------|---------------------|
| **Accuracy**| 0.56      | 0.53        | 0.49                |

# Visualization

