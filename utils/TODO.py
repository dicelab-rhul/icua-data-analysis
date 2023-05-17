
def correct_eyetracker_time(dataset, t):
    # Find the event preceeding the first eyetracking event, use this as the eyetracking timestamp. 
    # This will be accurate to 0.01 seconds (more than enough)
    ts = np.array(LineData.pack_variables(dataset, "timestamp")).astype(np.float32).squeeze()
    ti = np.arange(ts.shape[0])[ts == t[0]]
    pe = ts[ti-1] # previous event
    print(pe)
    print(dataset[ti.item()-1])
    
    #t = t - t[0] # eyetracker time needs to be converted into the global time

correct_eyetracker_time(dataset, t)

print(f"start: {datetime.fromtimestamp(start_time)}, finish: {datetime.fromtimestamp(finish_time)}, duration: {finish_time-start_time:2.0f}s")

print(f"start: {datetime.fromtimestamp(_start_time)}, finish: {datetime.fromtimestamp(_finish_time)}, duration: {_finish_time-_start_time:2.0f}s")

# how long was the user gazeing at each component?

print(t)

# proportion of gaze vs. saccade
#compute_time_intervals(i, t, )
