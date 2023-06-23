# icua-data-analysis
### TODO


exhaustive combinations of when people look when they act when it fails - did they look back after looking at a highlight?

compare subsequences in categorical sequences - e.g. they are on the fuel task, highlight shows on system, is the sequence of actions the same? 

arrow pointing to the task (throw away saccades etc). 

differences in best and worst performers, are there any differences in the way they act in the exhaustive combinations?





Q: Do participants improve over time? 
A: Hard to say without more trials... there doesn't appear to be significant trend. (see demographics.ipynb)

Q: Participants who said they did better, did they actually?
A: For the most part yes (see demographics.ipynb)

Q: Do participants look at one task while solving/interacting with another? 
A: Participants tend to look at the task that they are solving, keyboard/mouse input for a task is almost always done when looking at the task.

Q: What performance metrics might we use? 
A: 
    1. proportion of total time that there is a failure.
    2. average length of failures.
    3. number of failures.
    4. ... ???

### NOTES

1. The tracking agent was working with a flat 50 pixel L2 distance from the center rather than in the dotted box at the center of the task. This means it may not have been clear to the participants when the task was in error. @Szonya, please review the instructions for this task, was it to keep the target in the box? or close to the center?

2. The tracking task seems to easy, it is hardly ever (if ever) in error for most participants.

3. only 17 participants have proper eyetracking data, make it more clear in future experiments if the eyetracker is not correctly setup. 

4. for calibration purposes, ICUa v2 should also record the screen.

5. participant 18 icuaA eye tracker seems to have crashed halfway through.

6. since the arrow and the highlights show at the same time, its hard to know if they are using one or the other... we cant really know if they are following the arrow. An experiment is needed with different combinations of guidance.

### CHECKLIST 

- Implemented a parser for the event log files that can sort/extract data easily.
- Statistics on "gaze time" for each task/participant have been collected.
- Statistics on "time in error" for each task/participant have been collected.
- Statistics on "warning/highlight time" for each task/participant have been collected.
- Plots for all of the above "intervals"
- Plots for eye tracking data
- Plots for tracking task (sanity check for in/out of failure) and to see if there are any patterns.

#### Summary

- We have statistics for individuals, it is now time to compare individuals / work out some group statistics.
- Using the group statistics (or processed data (e.g. intervals)) we should work out some performance metrics to use for comparison.
- Try to draw some numerical links between user interaction (eye/mouse) and highlights. Data is ready for this step, but it is not entirely clear how to do it. The "warning" intervals tell us something about each participant reaction to agent feedback without much further analysis? Otherwise, comparing the "time in error" intervals for icu and icua WITH and WITHOUT feedback may be sufficient - we are interested mostly in the times where feedback is given...

