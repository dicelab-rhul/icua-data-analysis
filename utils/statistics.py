import pandas as pd
import numpy as np
from .data import load_nested_dict, merge_intervals, compute_time_intervals, ALL_WINDOW_PROPERTIES

class Statistics: 
        
    @classmethod
    def compute_failure_proportion_statistics(cls):
        data = load_nested_dict('data/Processed')
        def statistic(intervals, start_time, finish_time):
            # computes the total proportion of 
            dt = intervals[:,1] - intervals[:,0]
            return dt.sum() / (finish_time - start_time)
        df = pd.DataFrame(columns=['participant', 'experiment', *sorted(ALL_WINDOW_PROPERTIES.keys()), 'total'])
        for participant, _data1 in data.items():
            for experiment, _data2 in _data1.items():
                start_time, finish_time =  _data2['start_time'], _data2['finish_time']
                stats, all_failure_intervals = [], []
                for task, _data3 in sorted(_data2['tasks'].items()):
                    failure_intervals = [compute_time_intervals(x.failure, x.timestamp, start_time, finish_time).intervals for x in _data3.values()]
                    failure_intervals = merge_intervals(failure_intervals) # merge them to get failures for a single task
                    stats.append(statistic(failure_intervals, start_time, finish_time))
                    all_failure_intervals.append(failure_intervals)
                total_failure = merge_intervals(all_failure_intervals) # overlapping total
                total_stat = statistic(total_failure, start_time, finish_time)
                df.loc[len(df)] = [participant, experiment[3:], *stats, total_stat]
        return df

    @classmethod
    def compute_failure_length_statistics(cls):
        data = load_nested_dict('data/Processed')
        def statistic(intervals,):
            dt = intervals[:,1] - intervals[:,0]
            if dt.shape[0] == 0:
                return 0. # if there are no intervals, then the mean length is 0
            else :
                return dt.mean()

        df = pd.DataFrame(columns=['participant', 'experiment', *sorted(ALL_WINDOW_PROPERTIES.keys()), 'total'])
        for participant, _data1 in data.items():
            for experiment, _data2 in _data1.items():
                start_time, finish_time =  _data2['start_time'], _data2['finish_time']
                stats, all_failure_intervals = [], []
                for task, _data3 in sorted(_data2['tasks'].items()):
                    failure_intervals = [compute_time_intervals(x.failure, x.timestamp, start_time, finish_time).intervals for x in _data3.values()]
                    failure_intervals_m = merge_intervals(failure_intervals) # merge them to get failures for a single task
                    stats.append(statistic(failure_intervals_m))
                    all_failure_intervals.extend(failure_intervals)
                #total_failure = merge_intervals(all_failure_intervals) # overlapping total
                #total_stat = statistic(total_failure, start_time, finish_time)
                total_stat = statistic(np.concatenate(all_failure_intervals, axis=0))
                df.loc[len(df)] = [participant, experiment[3:], *stats, total_stat]
        return df

    @classmethod
    def compute_failure_count_statistics(cls): # computed slightly differently on a per-task basis.
        data = load_nested_dict('data/Processed')
        def statistic(intervals):
            return intervals.shape[0]
        df = pd.DataFrame(columns=['participant', 'experiment', *sorted(ALL_WINDOW_PROPERTIES.keys()), 'total'])
        for participant, _data1 in data.items():
            for experiment, _data2 in _data1.items():
                start_time, finish_time =  _data2['start_time'], _data2['finish_time']
                stats = []
                for task, _data3 in sorted(_data2['tasks'].items()):
                    failure_intervals = [compute_time_intervals(x.failure, x.timestamp, start_time, finish_time).intervals for x in _data3.values()]
                    counts = sum([statistic(fi) for fi in failure_intervals])
                    stats.append(counts)
                total_stat = sum(stats)
                df.loc[len(df)] = [participant, experiment[3:], *stats, total_stat]
        return df
