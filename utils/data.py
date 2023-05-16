
import pathlib 
import zipfile
import os
from pprint import pprint
from tqdm import tqdm
import shutil
import itertools
import more_itertools
import numpy as np


import re
from dataclasses import dataclass, asdict

SCALE_NAMES = ["Scale:0", "Scale:1", "Scale:2", "Scale:3"]
SCALE_NORMAL_STATE = 5
WARNINGLIGHT_NAMES = ["WarningLight:0", "WarningLight:1"]


DEFAULT_DIRECTORY = "./data/"
DATA_OWNER = "dicelab-rhul"
DATA_REPO = "icua-data-analysis"
DATA_RELEASE_TAG = "v1.0.0-data"
DATA_NAME = "ICUdata.zip"

# computes the time spent in failure for a particular task
# 'fail' should be a binary numpy array with 1's for each event that represents a failure (e.g. the moment a warning light or scale switches to the wrong state)
# 'timestamps' should be a numpy array of time stamps for each event
def compute_time_in_failure(fail, timestamps, start_time, finish_time):
    fail = np.pad(fail.astype(np.uint8), (1,1)) # pad with zeros either side (ensures even index cardinality)
    timestamps = np.pad(timestamps, (1,1))      # pad with start/end time
    timestamps[0] = start_time
    timestamps[-1] = finish_time
    y = np.pad(np.logical_xor(fail[:-1], fail[1:]), (1,0))
    yi = np.arange(y.shape[0])[y]
    ts = timestamps[yi].reshape(-1,2)
    df = ts[:,1] - ts[:,0]
    return dict(failure_intervals=ts, failure_proportion=df.sum() / (finish_time - start_time), 
                timestamps=timestamps.copy(), failures=fail.copy())

def create_datasets(force=False, n=None):
    # download(?) and generate a dataset from a given experiment log file. Each dataset is a collection of LineData objects.
    path = _get_dataset(force)
    files = list(sorted([f for f in path.iterdir() if f.suffix == ".txt"], key=lambda f: f.name))
    def data_generator(file, slice=None):
        with open(file, 'r') as f:
            try: 
                for line in itertools.islice(f, slice):
                    yield LineData.from_line(line)
            except Exception as e: 
                raise ValueError(f"Failed to parse file: {file}", e)
    dataset = dict()
    for file in tqdm(itertools.islice(files, n), desc="loading files..."):
        dataset[file.name] = [line for line in data_generator(file)]
    return dataset

def get_system_monitor_task_data(line_data):
    start_time = LineData.get_start_time(line_data)
    finish_time = LineData.get_finish_time(line_data)
    wl_data = [np.array(LineData.pack_variables(LineData.findall_from_source(line_data, wl), "timestamp", "value")) for wl in WARNINGLIGHT_NAMES]
    sc_data = [np.array(LineData.pack_variables(LineData.findall_from_source(line_data, sc), "timestamp", "value")) for sc in SCALE_NAMES]
    data = wl_data + sc_data
    #timestamp, value, fail
    for i in range(len(data)):
        data[i] = np.concatenate((data[i], np.zeros((data[i].shape[0], 1))),axis=-1)
    # compute failure cases
    for x in data[2:]: # for scales != 5
        x[:,2] = (x[:,1] != SCALE_NORMAL_STATE) # in failure
    data[0][:,2] = (data[0][:,1] == 0) # in failure
    data[1][:,2] = (data[1][:,1] == 1) # in failure
    return dict(components={k:v for k,v in zip(WARNINGLIGHT_NAMES + SCALE_NAMES, data)}, start_time=start_time, finish_time=finish_time)



@dataclass
class LineData:
    indx: int
    timestamp: float
    event_src: str
    event_dst: str
    variables: dict

    @classmethod
    def from_line(cls, line):
        pattern = r"^(.*?):(\d+\.\d+) - \((.*?)\): ({[^}]*('cause':None|\s*({[^}]*}|'[^']*'))[^}]*})$"
        match = re.match(pattern, line)
        if match:
            indx = int(match.group(1))
            timestamp = float(match.group(2))
            event_src, event_dst = match.group(3).split("->")
            variables = match.group(4) # this group is a bit dodgy because of the 'cause' variable... it should be removed.
            variables = variables.split("'cause'")[0] # this assumes that 'cause' is at the end... TODO CHECK THIS! its very annoying to process it otherwise
            if not variables.endswith("}"):
                variables += "}"
            variables = eval(variables)
            return cls(indx, timestamp, event_src, event_dst, variables)
        else:
            raise ValueError(f"Line:\n '{line}'\n   does not match pattern")

    @classmethod
    def get_start_time(cls, data):
        return data[0].timestamp

    @classmethod
    def get_finish_time(cls, data):
        return data[-1].timestamp

    @classmethod
    def findall_from_source(cls, data, event_src):
        # search through all "lines" to find all eyetracking events
        return list(filter(lambda x: x.event_src == event_src, data))
    
    @classmethod
    def in_from_source(cls, data, event_src):
        try:
            next(filter(lambda x: x.event_src == event_src, data)) # exception if empty
            return True
        except:
            return False

    @classmethod
    def pack_variables(cls, data, *keys):
        result = []
        for line in data:
            result.append([line.variables.get(k, asdict(line).get(k, None)) for k in keys])
        return result


# pull data from github and save it locally
def _get_dataset(force=False):
    directory = pathlib.Path(DEFAULT_DIRECTORY).resolve()
    if not directory.exists():
        directory.mkdir(parents=False)

    clean_directory = pathlib.Path(directory, DATA_NAME.split(".")[0]) # where data will end up
    if not clean_directory.exists():
        clean_directory.mkdir(parents=False)

    if force or len([f for f in clean_directory.iterdir()]) == 0:
        data_directory = pathlib.Path(directory, DATA_NAME.split(".")[0] + "-temp")
        if not data_directory.exists():
            data_directory.mkdir(parents=False)

        # pull data from github release
        url = f"https://github.com/{DATA_OWNER}/{DATA_REPO}/releases/download/{DATA_RELEASE_TAG}/{DATA_NAME}"
        os.system(f"wget -c --read-timeout=5 --tries=0 \"{url}\" -P {str(directory)}")
    
        with zipfile.ZipFile(str(pathlib.Path(directory, DATA_NAME)), 'r') as zip_ref:
            zip_ref.extractall(data_directory) # extract .zip data to this path
    
        # cleanup file names etc.
        def clean_file_name(f): # the file names are all over the place....?
            new_name = f.name.replace("_", "").replace("¬","")\
                    .replace(" ","").replace("hard","B")\
                    .replace("easy","A").split("-")[-1]\
                    .replace("p","P").replace("11x", "P11").replace("01x", "01")
            return pathlib.Path(f.parent.parent, DATA_NAME.split(".")[0],  new_name)

        files = [(f, clean_file_name(f)) for f in pathlib.Path(data_directory).iterdir() if f.name.endswith(".txt") and "event_log" not in f.name]
        files = list(sorted(files, key=lambda x : x[1]))
        old_directory = files[0][0].parent

        
        # copy files to new directory, this will be the working data directory.
        for f, fc in tqdm(files, desc="Copying and cleaning files..."):
            shutil.copyfile(str(f), str(fc))

        shutil.copyfile(pathlib.Path(old_directory, "A_config.json"), pathlib.Path(clean_directory, "A_config.json"))
        shutil.copyfile(pathlib.Path(old_directory, "B_config.json"), pathlib.Path(clean_directory, "B_config.json"))
        demo_name = "Participant Demographics (red shoes-cockpit).xlsx"
        shutil.copyfile(pathlib.Path(old_directory, demo_name), pathlib.Path(clean_directory, "demographics.xlsx"))
        shutil.rmtree(data_directory)
    return clean_directory
    