import json
import os
import re


def prepare_vqa_dataset(image_path, questions_path, annotations_path, valid_answer_set, data_type="train2014"):
    images_folder_base = image_path + "/%s/"
    image_file_base = 'COCO_'+data_type+'_%012d'
    questions_file_base = questions_path + '/v2_OpenEnded_mscoco_%s_questions.json'
    annotations_file_base = annotations_path + '/v2_mscoco_%s_annotations.json'

    # prepare question to annotations
    annotations = load_json_file(annotations_file_base % data_type)["annotations"]
    question_to_annotation_dict = {a["question_id"]:a for a in annotations}

    # prepare questions
    questions = load_json_file(questions_file_base % data_type)["questions"]

    # prepare images
    images_folder = (images_folder_base % data_type)

    # dataset list
    dataset = []

    # prepare question, answer, image mapping
    for q in questions:
        question_id = q["question_id"]
        image_id = q["image_id"]
        image_name = os.path.join(images_folder, image_file_base % image_id+".jpg")
        question_string = q["question"]
        question_token = tokenize(question_string)

        answer_annotation = question_to_annotation_dict[question_id]
        answers_all, valid_answers = extract_answers(answer_annotation['answers'], valid_answer_set)
        if len(valid_answers) == 0:
                valid_answers = ['<unk>']

        data = dict(question_id= question_id,
                    image_id= image_id,
                    image_name= image_name,
                    question_string= question_string,
                    question_token= question_token,
                    answers_all= answers_all,
                    valid_answers= valid_answers)

        dataset.append(data)

    return dataset

def load_json_file(filename):
    with open(filename, 'r') as f:
        json_input = json.load(f)
    return json_input 

def tokenize(sentence):
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    tokens = SENTENCE_SPLIT_REGEX.split(sentence.lower())
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens

def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines

def extract_answers(q_answers, valid_answer_set):
    all_answers = [answer["answer"] for answer in q_answers]
    valid_answers = [a for a in all_answers if a in valid_answer_set]
    return all_answers, valid_answers