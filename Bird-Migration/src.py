import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

birddata = pd.read_csv("bird_tracking.csv")

timeobjects = []
for timedates in birddata.date_time:
    timeobjects.append(datetime.datetime.strptime(timedates[:-3], "%Y-%m-%d %H:%M:%S"))
timeobjects[:10]

birddata["timestamps"] = pd.Series(timeobjects, index = birddata.index)


# For Eric

EricTime = birddata.timestamps[birddata.bird_name == 'Eric']
elapsed_time = [time - EricTime[0] for time in EricTime]

# Plot
plt.plot(np.array(elapsed_time)/datetime.timedelta(days = 1))
plt.axis("tight")
plt.xlabel("Observation")
plt.ylabel("No. of Days");