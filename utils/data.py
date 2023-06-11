
import pathlib 
import zipfile
import os
import shutil
import itertools
import more_itertools
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import re
import pandas as pd
import json

from types import SimpleNamespace
from cycler import cycler
from pprint import pprint
from tqdm import tqdm


from dataclasses import dataclass, asdict

# valid states from config files (TODO CHANGE IF THE EXPERIMENTS CHANGE)
TARGET_DISTANCE = 50
SCALE_NORMAL_STATE = 5
FUEL_CAPACITY = 2000
FUEL_ACCEPTABLE_RANGE = (FUEL_CAPACITY/2-FUEL_CAPACITY*0.3,FUEL_CAPACITY/2+FUEL_CAPACITY*0.3)

# widget names
SCALE_NAMES = ["Scale:0", "Scale:1", "Scale:2", "Scale:3"]
WARNINGLIGHT_NAMES = ["WarningLight:0", "WarningLight:1"]
FUEL_TANK_NAMES = ["FuelTank:A", "FuelTank:B"]

EYETRACKER_NAME = "EyeTracker:0"
EYETRACKERSTUB_NAME = "EyeTrackerStub"
TARGET_NAME = "Target:0"
ARROW_NAME = "arrow_rotator_TEST"

# data loading
DEFAULT_DIRECTORY = "./data/"
DATA_OWNER = "dicelab-rhul"
DATA_REPO = "icua-data-analysis"
DATA_RELEASE_TAG = "v1.0.0-data"
DATA_NAME = "ICUdata.zip"
DEMO_FILE = "ICUData/demographics.xlsx"

# window properties
WINDOW_SIZE = (800,800)
TRACKING_WINDOW_PROPERTIES = {'position': np.array((351.25, 37.85)), 'size': np.array((341.25, 341.25)), 'color':'#4363d8', 'warning_name':'Highlight:TrackingMonitor', 'data_fn':lambda dataset:get_tracking_task_data(dataset) }
FUEL_WINDOW_PROPERTIES = {'position': np.array((253.75, 455.71428571428567)), 'size': np.array((536.25, 334.2857142857143)), 'color':'#3cb44b', 'warning_name':'Highlight:FuelMonitor','data_fn':lambda dataset:get_fuel_task_data(dataset)  }
SYSTEM_WINDOW_PROPERTIES = {'position': np.array((10.0, 37.857142857142854)), 'size': np.array((243.75, 390.0)), 'color':'#e6194B', 'warning_name':'Highlight:SystemMonitor', 'data_fn':lambda dataset: get_system_monitor_task_data(dataset) }
ALL_WINDOW_PROPERTIES = {'system':SYSTEM_WINDOW_PROPERTIES, 'fuel':FUEL_WINDOW_PROPERTIES, 'tracking':TRACKING_WINDOW_PROPERTIES}

# plotting
ICU_BACKGROUND_IMAGE = mpimg.imread('./results/images/background.png') # background image
COLOR_CYCLE = cycler(color=['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000'])
plt.rcParams['axes.prop_cycle'] = COLOR_CYCLE

# get time stamp of each click event grouped by the clicked component
def get_clicks(line_data):
    line_data = LineData.findall_from_src(line_data, "Canvas") # will contain all click events
    return {k:np.array(LineData.pack_variables(v, "timestamp")) for k,v in LineData.groupby_dst(line_data)}
    
class Statistics:

    @classmethod
    def compute_failure_proportion(cls, intervals, start_time, finish_time):
        # computes the total proportion of 
        dt = intervals[:,1] - intervals[:,0]
        return dt.sum() / (finish_time - start_time)

    @classmethod 
    def compute_mean_interval_length(cls, intervals, *args):
        dt = intervals[:,1] - intervals[:,0]
        if dt.shape[0] == 0:
            return 0. # if there are no intervals, then the mean length is 0
        else :
            return dt.mean()

    @classmethod
    def compute_failure_count(cls, intervals, *args):
        return intervals.shape[0]
    
# numpy array of time intervals to merge overlapping
def merge_intervals(intervals):
    assert all([y.shape[-1] == 2 for y in intervals])
    ts = np.concatenate(intervals, axis=0)
    ts = ts[np.argsort(ts[:,0])] # sort by start times
    s, f = ts[:,0], np.maximum.accumulate(ts[:,1])
    v = np.ones(ts.shape[0] + 1, dtype=bool)
    v[1:-1] = s[1:] >= f[:-1]
    s, f = s[v[:-1]], f[v[1:]]
    return np.vstack([s,f]).T

# check if coordinate (x,y) are in a box at position 'pos' (top-left) and 'size'
def in_box(x, y, pos, size):
    interval_x = (pos[0], pos[0] + size[0])
    interval_y = (pos[1], pos[1] + size[1])
    xok = np.logical_and(x > interval_x[0], x < interval_x[1])
    yok = np.logical_and(y > interval_y[0], y < interval_y[1])
    return np.logical_and(xok, yok)

# computes the number of groups of consequtive ones in a binary array. [0,1,1,1,0,0,1,1,0,1] would be 3 for example.
def compute_num_groups_of_ones(binary):
    return int((np.diff(binary) < 0).sum() + binary[-1])

# computes the time spent in failure for a particular task
# 'binary' should be a binary numpy array with 1's for each event that represents a given state (e.g. the moment a warning light or scale switches to the wrong state)
# 'timestamps' should be a numpy array of time stamps for each event
def compute_time_intervals(binary, timestamps, start_time, finish_time, pad='finish'):
    binary = np.pad(binary.astype(np.uint8), (1,1)) # pad with zeros either side (ensures even index cardinality)
    timestamps = np.pad(timestamps, (1,1))          # pad with start/end time
    timestamps[0] = start_time
    timestamps[-1] = finish_time if pad == 'finish' else timestamps[-2] # otherwise pad with the current value...
    y = np.pad(np.logical_xor(binary[:-1], binary[1:]), (1,0))
    yi = np.arange(y.shape[0])[y]
    ts = timestamps[yi].reshape(-1,2)
    df = ts[:,1] - ts[:,0]
    return SimpleNamespace(**dict(intervals=ts, proportion=df.sum() / (finish_time - start_time), 
                timestamps=timestamps.copy(), binary=binary.copy()))

def get_clean_datasets(force=False, n=None):
    n = n * 4 if n is not None else None
    datasets = create_datasets(force=force, n=n)
    # prune dataset to contain only those with valid eyetracking data...
    datasets = {k:data for k, data in datasets.items() if LineData.contains_src(data, EYETRACKER_NAME)}
    datasets = list(sorted([(k.split(".")[0],v) for k,v in datasets.items()], key=lambda x: x[0][:3]))
    return {k:dict(g) for k,g in itertools.groupby(datasets, key=lambda x: x[0][:3])}

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
        # fix eye tracking timestamps... this is annoying but it is what it is..
        eye_first_timestamp = None
        eye_1prev_timestamp = None
        for i, line in enumerate(dataset[file.name]):
            if line.event_src == EYETRACKER_NAME: # first occurance of eyetracker data. 
                eye_first_timestamp = line.timestamp
                eye_1prev_timestamp = dataset[file.name][i-1].timestamp # i should not be 0...
                break
        for line in LineData.findall_from_src(dataset[file.name], EYETRACKER_NAME):
            line.timestamp = (line.timestamp - eye_first_timestamp) + eye_1prev_timestamp    
    return dataset

def get_system_monitor_task_data(line_data):
    start_time = LineData.get_start_time(line_data)
    finish_time = LineData.get_finish_time(line_data)
    wl_data = [np.array(LineData.pack_variables(LineData.findall_from_src(line_data, wl), "timestamp", "value")) for wl in WARNINGLIGHT_NAMES]
    sc_data = [np.array(LineData.pack_variables(LineData.findall_from_src(line_data, sc), "timestamp", "value")) for sc in SCALE_NAMES]
    data = wl_data + sc_data
    #timestamp, value, fail
    for i in range(len(data)):
        data[i] = np.concatenate((data[i], np.zeros((data[i].shape[0], 1))),axis=-1)
    # compute failure cases
    for x in data[2:]: # for scales != 5
        x[:,2] = (x[:,1] != SCALE_NORMAL_STATE) # in failure
    data[0][:,2] = (data[0][:,1] == 0) # in failure
    data[1][:,2] = (data[1][:,1] == 1) # in failure
    return SimpleNamespace(**dict(components={k:pd.DataFrame(dict(timestamp=v[:,0], value=v[:,1], failure=v[:,2].astype(bool))) for k,v in zip(WARNINGLIGHT_NAMES + SCALE_NAMES, data)}, start_time=start_time, finish_time=finish_time))

def get_tracking_task_data(line_data, l2_threshold=50):
    start_time = LineData.get_start_time(line_data)
    finish_time = LineData.get_finish_time(line_data)
    def failure_l2(x, y, l2_threshold=50): 
        return np.sqrt(x ** 2 + y ** 2) > l2_threshold
    target_data = LineData.findall_from_src(line_data, TARGET_NAME)
    target_data = LineData.pack_variables(target_data, "timestamp", "x", "y")
    target_data = np.array(target_data)
    fail = failure_l2(target_data[:,1], target_data[:,2], l2_threshold=l2_threshold)  
    df = pd.DataFrame(dict(timestamp=target_data[:,0], x=target_data[:,1], y=target_data[:,2], failure=fail))
    return SimpleNamespace(components={TARGET_NAME:df}, start_time=start_time, finish_time=finish_time)
    

def get_fuel_task_data(line_data):
    start_time = LineData.get_start_time(line_data)
    finish_time = LineData.get_finish_time(line_data)
    def _get_data(src, line_data):
        line_data = LineData.findall_from_src(line_data, src)
        line_data = LineData.findall_from_key_value(line_data, "label", "fuel")
        data = np.array(LineData.pack_variables(line_data, "timestamp", "acceptable"))
        # TODO include the tanks value? this will need to be obtained from other events (label:change)
        return pd.DataFrame(dict(timestamp=data[:,0], failure=1-data[:,1].astype(bool)))
    tank_data = {src:_get_data(src, line_data) for src in FUEL_TANK_NAMES}
    return SimpleNamespace(components=tank_data, start_time=start_time, finish_time=finish_time)
    


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
    def findall_from_src(cls, data, event_src):
        # search through all "lines" to find all eyetracking events
        return list(filter(lambda x: x.event_src == event_src, data))
    
    @classmethod
    def findall_from_dst(cls, data, event_dst):
        # search through all "lines" to find all eyetracking events
        return list(filter(lambda x: x.event_dst == event_dst, data))
    
    @classmethod
    def findall_from_key_value(cls, data, key, value):
        return list(filter(lambda x: x.variables[key] == value, data))

    @classmethod
    def groupby_src(cls, data):
        return {k:list(sorted(v, key=lambda x:x.timestamp)) for k,v in itertools.groupby(sorted(data, key=lambda x: x.event_src), key=lambda x: x.event_src)}
        
    @classmethod
    def groupby_dst(cls, data):
        return {k:list(sorted(v, key=lambda x:x.timestamp)) for k,v in itertools.groupby(sorted(data, key=lambda x: x.event_dst), key=lambda x: x.event_dst)}

    @classmethod
    def contains_src(cls, data, event_src):
        try:
            next(filter(lambda x: x.event_src == event_src, data)) # exception if empty
            return True
        except:
            return False

    @classmethod
    def pack_variables(cls, data, *keys, sort_by='timestamp'):
        result = []
        sort_by = keys.index(sort_by)
        for line in data:
            result.append([line.variables.get(k, asdict(line).get(k, None)) for k in keys])
        return list(sorted(result, key=lambda v : v[sort_by]))

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
    

def get_eyetracking_data(dataset):
    eye_data = LineData.pack_variables(LineData.findall_from_src(dataset, EYETRACKER_NAME), "timestamp", "label", "x", "y")
    eye_data = np.array(eye_data)
    gi = (eye_data[:,1] == "gaze").astype(bool) # gaze = 1, saccade = 0
    t, x, y = eye_data[:,0].astype(np.float64), eye_data[:,2].astype(np.float32), eye_data[:,3].astype(np.float32)
    return pd.DataFrame(data=dict(timestamp=t,x=x,y=y,gaze=gi))

def get_warning_data(dataset):
    finish_time = LineData.get_finish_time(dataset)
    def get_data_from_source(src):
        data = np.array(LineData.pack_variables(LineData.findall_from_src(dataset, src), "timestamp", "value"))
        data = data.reshape(data.shape[0], 2) # in case there are no events
        if data.shape[0] % 2 != 0:
            # the session ended with a warning... add another event to match it (turn off at the end of session)
            data = np.concatenate([data, np.zeros((1,2))])
            data[-1,0] = finish_time
        return pd.DataFrame(dict(timestamp=data[:,0], value=data[:,1].astype(bool)))
    # dataframes instead?
    return {k:get_data_from_source(v['warning_name']) for k,v in ALL_WINDOW_PROPERTIES.items()}

def get_task_data(dataset):
    return {task:properties['data_fn'](dataset).components for task, properties in ALL_WINDOW_PROPERTIES.items()}    
    
def get_demographics_data():
    df = pd.read_excel("./data/ICUdata/demographics.xlsx")
    df = df[['Participant Number', 'Easy icu A', 'Easy icua A', 'Hard icu B', 'Hard icua B']]
    df = df.rename(columns={"Participant Number": "participant", 'Easy icu A':0, 'Easy icua A':1, 'Hard icu B':2, 'Hard icua B':3})
    df = df.applymap(lambda x: x.replace(" ", "").replace("'", ""))
    return df

def get_arrow_data(dataset):
    line_data_arrow = LineData.findall_from_src(dataset, ARROW_NAME)
    data_arrow = np.array(LineData.pack_variables(line_data_arrow, "timestamp", "angle")).reshape(len(line_data_arrow),2)
    return pd.DataFrame(dict(timestamp=data_arrow[:,0], angle_delta=data_arrow[:,1], angle=data_arrow[:,1].cumsum()))

def get_keyboard_data(dataset):
    line_data_keyboard = LineData.findall_from_src(dataset, "KeyHandler")
    data_keyboard = np.array(LineData.pack_variables(line_data_keyboard, "timestamp", "key", "action"))
    return pd.DataFrame(dict(timestamp=data_keyboard[:,0], key=data_keyboard[:,1], action=data_keyboard[:,2]))
  
def get_mouse_data(dataset):
    line_data_mouse = LineData.findall_from_key_value(dataset, "label", "click")
    data_mouse = np.array(LineData.pack_variables(line_data_mouse, "timestamp", "event_dst", "x", "y"))
    df = pd.DataFrame(dict(timestamp=data_mouse[:,0].astype(np.float64), 
                             dst=data_mouse[:,1]))
    def get_task_from_component(x):
        if "Pump" in x:
            return "fuel"
        elif "WarningLight" in x or "Scale" in x:
            return "system"
        else:
            raise ValueError(f"Unexpected component {x}")
 
    df['task'] = df['dst'].apply(get_task_from_component)
    df['x'] = data_mouse[:,2].astype(int)
    df['y'] = data_mouse[:,3].astype(int)
    return df

# sanity check for window properties
#plt.figure()
#plt.imshow(img)
#plt.gca().add_patch(plt.Rectangle(TRACKING_WINDOW_PROPERTIES['position'], *TRACKING_WINDOW_PROPERTIES['size'], color='red', fill=False))
#plt.gca().add_patch(plt.Rectangle(FUEL_WINDOW_PROPERTIES['position'], *FUEL_WINDOW_PROPERTIES['size'], color='red', fill=False))
#plt.gca().add_patch(plt.Rectangle(SYSTEM_WINDOW_PROPERTIES['position'], *SYSTEM_WINDOW_PROPERTIES['size'], color='red', fill=False))


# save/load a dict of numpy arrays.

def save_nested_dict(data, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    meta_data = {}

    for key, value in data.items():
        if isinstance(value, dict):
            path = os.path.join(directory, str(key))
            save_nested_dict(value, path)
        elif isinstance(value, np.ndarray):
            np.savez_compressed(os.path.join(directory, f"{key}.npz"), value=value)
        elif isinstance(value, pd.DataFrame):  # Handle DataFrame objects
            value.to_csv(os.path.join(directory, f"{key}.csv"), index=False)
        else:
            meta_data[key] = value

    if meta_data:
        with open(os.path.join(directory, "meta.json"), "w") as file:
            json.dump(meta_data, file)

def load_nested_dict(directory):
    data = {}

    meta_path = os.path.join(directory, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as file:
            meta_data = json.load(file)
            data.update(meta_data)

    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            data[item] = load_nested_dict(path)
        elif item.endswith('.npz'):
            key = item[:-4]  # Remove '.npz' extension
            npz_data = np.load(path)
            data[key] = npz_data['value']
        elif item.endswith('.csv'):  # Handle DataFrame files
            key = item[:-4]  # Remove '.csv' extension
            data[key] = pd.read_csv(path)

    return dict(sorted(data.items()))