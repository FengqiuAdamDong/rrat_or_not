import numpy as np
import json
from matplotlib import pyplot as plt
class injectionsData:
    #each instance contains an injection detection pair - if not detected, detection_dict is None
    def __init__(self, injection_dict, detection_dict=None):
        #may sort this out later, for now leave as is
        self.injection = injection_dict
        self.detection = detection_dict


def load_injections_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    injections_list = data.get('injections', [])
    detections_list = data.get('detections', [])
    injections_id_list = [inj['id'] for inj in injections_list]
    detections_id_list = [det['det_id'] for det in detections_list]
    injections_id_list = np.array(injections_id_list)
    detections_id_list = np.array(detections_id_list)
    #match detections to injections
    injections_data_arr = []
    for i,injections_id in enumerate(injections_id_list):
        index = np.where(injections_id == detections_id_list)[0]
        #if no match, add injectionsdata
        if len(index) == 0:
            inj_obj = injectionsData(injection_dict=injections_list[i], detection_dict=None)
            injections_data_arr.append(inj_obj)
            continue
        if len(index) > 1:
            print(f"Warning: Multiple detections found for injection id {injections_id}. Using the first one.")
            import pdb; pdb.set_trace()
        det_index = index[0]
        inj_obj = injectionsData(injection_dict=injections_list[i], detection_dict=detections_list[det_index])
        injections_data_arr.append(inj_obj)
    #save this array or do something with it later
    np.save('injections_data_arr.npy', injections_data_arr, allow_pickle=True)
    #get the dm and plot them


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load injections data from a JSON file.")
    parser.add_argument('file_path', type=str, help='Path to the injections JSON file.')
    args = parser.parse_args()
    load_injections_data(args.file_path)
