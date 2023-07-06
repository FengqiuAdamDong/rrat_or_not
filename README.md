# rrat_or_not

Here I will document how to run injections for CHIME/Pulsar filterbank files
* Detections *
You need the detections first to find the width\\
First run CHIPSPIPE so that you have the `positive_bursts_1` directory
- run `python ~/Documents/rrat_or_not/utils/filter_similar_bursts.py -folder_path positive_bursts_1/ -dm $DM` to get a `filtered` folder and csv, this removed duplicate bursts
- remove the rfi
- run `python ~/Documents/rrat_or_not/utils/create_positive_csv_edit.py filtered.csv` this gives you `filtered_edit.csv` which is the list of bursts you should find the snr of
- run `python ~/Documents/rrat_or_not/injection_scripts_fluence/extract_snr.py -dm 30.996 -o B1905+39 -ds 3 -p filtered_edit.csv` to extract snr (you may need to do a `cp */*rfifind* .` to get all the rfifind files to the directory you're working in
- 



*Injections*
- run move_injection_files.py
`python ~/rrat_or_not/injection_scripts_fluence/move_injections_files.py --path injections --gap 50`
- run multiple_cut.sh
`~/rrat_or_not/utils/multiple_cut.sh *.fil`
- run fold_all_filterbank.py
- run multiple_inject.sh (use -l if running locally)
- run mutlple_detect.sh (use -l if runnning locally)
- at the same time run run_check_single_pulse.sh (you will need the rfi statistics before doing that)
- python combine_inject_stats.py *.fil

You should get an overall curve that tells you how CHIPSPIPE has done
