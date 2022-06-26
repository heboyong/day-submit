import json
import numpy as np
import pickle
import tqdm


def main_json():
    image_json = 'data/test_submit.json'
    result_pkl = 'submit_last/result.pkl'
    output_pkl = "submit_last/predictions.pkl"


    months = ['Apr', 'Aug', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Sep']
    with open(result_pkl, 'rb') as result_:
        results = pickle.load(result_)
    with open(image_json, 'r') as image_json:
        images = json.load(image_json)

    image_dict = {}
    result_info = {}

    for month in months:
        result_info[month] = {}

    for image_message in images["images"]:
        image_dict[str(image_message['id'])] = str(image_message['file_name'])

    for image_id in tqdm.tqdm(range(len(images["images"]))):

        result = results[image_id]
        scores_list = []
        boxes_list = []
        labels_list = []
        for class_id in range(4):
            boxes = result[class_id]
            for box in boxes:
                scores_list.append(box[4])
                boxes_list.append(list(np.array(box[0:4], dtype=int)))
                labels_list.append(class_id)

        score_index = sorted(range(len(scores_list)), key=lambda k: scores_list[k], reverse=True)

        scores_list = np.array(scores_list)[score_index]
        labels_list = np.array(labels_list)[score_index]
        boxes_list = np.array(boxes_list)[score_index]

        out_bboxes = []
        out_labels = []

        '''
        human
        bicycle
        motorcycle
        vehicle
        '''

        for score, label, box in zip(scores_list, labels_list, boxes_list):
            if (label == 0 and score >= 0.95) or (label == 1 and score >= 0.5) or (
                    label == 2 and score >= 0.1) or (label == 3 and score >= 0.97):

                out_labels.append(label)
                out_bboxes.append(box)
        image_name = image_dict[str(image_id)].split('/')[-1]
        image_month = image_name[0:3]
        output_name = image_name[4:].split('.')[0]
        result_info[str(image_month)][output_name] = {}
        result_info[str(image_month)][output_name]["boxes"] = list(out_bboxes)
        result_info[str(image_month)][output_name]["labels"] = list(out_labels)

    with open(output_pkl, "wb") as f:
        pickle.dump(result_info, f)


if __name__ == '__main__':
    main_json()
