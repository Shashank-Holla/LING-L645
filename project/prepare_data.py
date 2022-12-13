import os
import glob
import shutil
import sys

# download questions file for training and test 
def download_and_extract_data(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    zipped_path = os.path.join(data_path, "zip")
    annotations_path = os.path.join(data_path, "annotations")
    questions_path = os.path.join(data_path, "questions")
    image_path = os.path.join(data_path, "images")

    # question set- train and validation set
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P '+zipped_path)
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P '+zipped_path)
    print("Questions files have been downloaded.")

    # annotations set
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P '+zipped_path)
    os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P '+zipped_path)
    print("Annotation files have been downloaded.")

    # image set
    os.system('wget http://images.cocodataset.org/zips/train2014.zip -P '+zipped_path)
    os.system('wget http://images.cocodataset.org/zips/val2014.zip -P '+zipped_path)
    print("Image files have been downloaded.")    

    # unzip
    for zipped_file in glob.glob(zipped_path+"/*"):
        if "Annotations" in zipped_file:
            destination_path = annotations_path
        elif "Questions" in zipped_file:
            destination_path = questions_path
        elif "2014" in zipped_file:
            destination_path = image_path

        os.system('unzip -q '+zipped_file+' -d '+destination_path)   

    print("Annotations, questions and images have been extracted.")

    shutil.rmtree(zipped_path)
    print("Removed zip files.")


if __name__ == "__main__":
    data_path = sys.argv[1]
    download_and_extract_data(data_path)