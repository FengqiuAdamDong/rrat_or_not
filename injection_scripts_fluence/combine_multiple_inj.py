import sys
from inject_stats import inject_stats
import dill
if __name__ == "__main__":
    file_list = sys.argv[1:]
    #load all the inject stats files
    inj_stats_arr = []
    for f in file_list:
        with open(f, 'rb') as inj_stats_file:
            inj_stats_arr.append(dill.load(inj_stats_file))
    #combine them
    for i,inj_stat in enumerate(inj_stats_arr):
        if i == 0:
            combined_inj_stats = inj_stat
        else:
            combined_inj_stats.combine_inj_stats(inj_stat)
    #save the combined inject stats
    with open('inj_stats.dill', 'wb') as inj_stats_file:
        dill.dump(combined_inj_stats, inj_stats_file)
