# rrat_or_not

Here I will document how to run injections for CHIME/Pulsar filterbank files
*Injections*
First off,
Run check_single_pulse.sh from the automated filterbank repo, you'll need the mask files
Then run multiple_inject.sh, you can alter the inject parameters in injec_pulses.py
Next run run_check_single_pulse.sh to run the pipeline on all the injected filterbank files
Finally run multiple_detect.sh to get detection statistics
