import subprocess
import argparse
from os.path import basename, split, join
import shutil
from glob import glob
from ml_reusable.utils.multiproc import run_in_parallel


def download_data(path):
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    cmd = ["wget", url, "-P", path]
    subprocess.run(cmd)


def extract(filename):
    path = split(filename)[0]
    cmd = ["tar", "xvjf", filename, "-C", path]
    subprocess.run(cmd)


def downsample(wav):
    sr = "16000"
    tmp = "/tmp/" + basename(wav)
    cmd = ["sox", wav, "-r", sr, tmp]
    subprocess.run(cmd)
    shutil.move(tmp, wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LJSpeech")
    parser.add_argument("-d", "--datapath", default="Data")
    parser.add_argument("-sr", "--sampling_rate", type=int, default=16000)
    args = parser.parse_args()

    print("----------Downloading Data------------")
    download_data(args.datapath)

    print("----------Extracting Data-------------")
    file = join(args.datapath, "LJSpeech-1.1.tar.bz2")
    extract(file)

    print("----------Downsampling Data-----------")
    print("==> ", args.sampling_rate)
    wpath = join(args.datapath, "LJSpeech-1.1/wavs", "*.wav")
    wavs = glob(wpath)
    run_in_parallel(downsample, wavs, "Downsampling audio")
