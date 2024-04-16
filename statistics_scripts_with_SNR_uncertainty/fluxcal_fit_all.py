import ephem
import numpy as np
import beam_model
import pytz
import datetime
from astropy.time import Time
import csv
from fluxcal_fit import fluxcal_fit
import glob
import shutil


def calculate_transit_time(rajd, decjd):
    """Calculate the CHIME transit time for a known source.
    Parameters
    ----------
    rajd : float
        Right ascension, in degrees.
    decjd : float
        Declination, in degrees.
    date : :obj:datetime
        Date to calculate transit time for.
    Returns
    -------
    transit_time : :obj:datetime
        UTC transit time of `source` on `date`.
    """
    # date = datetime.datetime.utcnow()
    date = datetime.datetime(2023, 12, 1)
    coord = ephem.Equatorial(ephem.degrees(str(rajd)), ephem.degrees(str(decjd)))
    body = ephem.FixedBody()
    body._ra = coord.ra
    body._dec = coord.dec

    pst = pytz.timezone("Canada/Pacific")
    date = date - pst.utcoffset(date)

    beam_model.config.chime.date = ephem.Date(date)
    transit_time = beam_model.config.chime.next_transit(body).datetime()

    # explicitly make this UTC time
    transit_time = transit_time.replace(tzinfo=pytz.utc)

    return transit_time


def read_pulsar_pop(filename):
    # read the pulsar population filename
    with open(filename, "r") as f:
        reader = csv.reader(f)
        pulsar_name = []
        rajd = []
        decjd = []

        for i, row in enumerate(reader):
            if i == 0:
                continue
            pulsar_name.append(row[6])
            rajd.append(float(row[8]))
            decjd.append(float(row[9]))
    pulsar_name = np.array(pulsar_name)
    rajd = np.array(rajd)
    decjd = np.array(decjd)
    return pulsar_name, rajd, decjd


def read_calibrator_list(filename):
    # read the pulsar population filename
    with open(filename, "r") as f:
        reader = csv.reader(f)
        calibrator_name = []
        rajd = []
        decjd = []

        for i, row in enumerate(reader):
            if i == 0:
                continue
            calibrator_name.append(row[0])
            rajd.append(float(row[2]))
            decjd.append(float(row[3]))
    calibrator_name = np.array(calibrator_name)
    rajd = np.array(rajd)
    decjd = np.array(decjd)
    return calibrator_name, rajd, decjd


if __name__ == "__main__":
    pulsar_name, rajd_pulsar, decjd_pulsar = read_pulsar_pop("pulsar_pop_sheet.csv")
    calibrator_name, rajd_cal, decjd_cal = read_calibrator_list(
        "flux_calibrator_list.csv"
    )
    for pulsar, ra_pulsar, dec_pulsar in zip(pulsar_name, rajd_pulsar, decjd_pulsar):
        # calculate the transit time for the pulsar
        transit_time = calculate_transit_time(ra_pulsar, dec_pulsar)
        # convert transit time to MJD
        transit_time = Time(transit_time).mjd
        # find the closest calibrator
        dec_calibrator_diff = np.abs(decjd_cal - dec_pulsar)
        ind_min = np.argmin(dec_calibrator_diff)
        cal_name = calibrator_name[ind_min]
        cal_ra = rajd_cal[ind_min]
        cal_dec = decjd_cal[ind_min]
        # glob the bayesian results file
        bayesian_results = glob.glob(f"{pulsar}*lnln_results.npz")
        try:
            fluxcal_fit(
                bayesian_results[0],
                cal_name,
                ra_pulsar,
                dec_pulsar,
                transit_time,
                cal_error=0.1535,
            )
        except:
            print(f"Error in fitting {pulsar}")
            pass
        # copy {pulsar}.yaml to a name with the calibrator
        try:
            shutil.copyfile(f"{pulsar}.yaml", f"{pulsar}_{cal_name}_calibrated.yaml")
        except:
            print(f"Error in copying {pulsar}.yaml")
            pass


# transit_time = calculate_transit_time(290.436729, 21.883958)
# convert transit time to MJD
# transit_time = Time(transit_time).mjd
# print(transit_time)
