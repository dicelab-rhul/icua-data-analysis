
import pathlib 
import zipfile
import os
from pprint import pprint
from tqdm import tqdm
import shutil
import itertools
import re
from dataclasses import dataclass, asdict

DEFAULT_DIRECTORY = "./data/"
DATA_OWNER = "dicelab-rhul"
DATA_REPO = "icua-data-analysis"
DATA_RELEASE_TAG = "v1.0.0-data"
DATA_NAME = "ICUdata.zip"

FILE_NAME_REGEX = ""



def create_dataset(force=False, n=None):
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
    def findall_from_source(cls, data, event_src):
        # search through all "lines" to find all eyetracking events
        return list(filter(lambda x: x.event_src == event_src, data))
    
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
    