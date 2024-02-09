# rrat_or_not

Here I will document how to run injections for CHIME/Pulsar filterbank files
* Detections *
The script to extract SNR is called `rrat_or_not/injection_scripts_fluence/batch_extract_snr.sh $DM $outputfn $filtered_edit.csv $period`

The script to submit lots of small individual injections for width/snr combo is called `inject_pulses_sigpyproc.py`

The script to detect pulses is called `multiple_detect_multiple_jobs.sh`

The script to check if all the injections are completed is called

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
