import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
dataset = pd.read_csv(r"C:\Users\MAHABHARAT\Desktop\ECG\FINAL PROJECT 8 SEM\data.csv")#read data 
#Calculate moving average with 0.75s in both directions, then append do dataset
hrw = 0.200 #One-sided window size, as proportion of the sampling frequency
fs = 250 #The example dataset was recorded at 250Hz
mov_avg = dataset['hart'].rolling(int(hrw*fs)).mean() #Calculate moving average #for older panda version mov_avg = pd.rolling_mean(dataset.hart, window=int(hrw*fs))
#Impute where moving average function returns NaN, which is the beginning of the signal where x hrw
avg_hr = (np.mean(dataset.hart))
mov_avg = [avg_hr if math.isnan(x) else x for x in mov_avg]
mov_avg = [x*1.2 for x in mov_avg] #For now we raise the average by 20% to prevent the secondary heart contraction from interfering
dataset['hart_rollingmean'] = mov_avg #Append the moving average to the dataframe

#Mark regions of interest
window = []
peaklist = []
listpos = 0 #We use a counter to move over the different data columns
for datapoint in dataset.hart:
    rollingmean = dataset.hart_rollingmean[listpos] #Get local mean
    if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
        listpos += 1
    elif (datapoint > rollingmean): #If signal comes above local mean, mark ROI
        window.append(datapoint)
        listpos += 1
    else: #If signal drops below local mean -> determine highest point
        maximum = max(window)
        beatposition = listpos - len(window) + (window.index(max(window))) #Notate the position of the point on the X-axis
        peaklist.append(beatposition) #Add detected peak to list
        window = [] #Clear marked ROI
        listpos += 1
ybeat = [dataset.hart[x] for x in peaklist] #Get the y-value of all peaks for plotting purposes
print (peaklist)

RR_list = []
cnt = 0
while (cnt < (len(peaklist)-1)):
    RR_interval = (peaklist[cnt+1] - peaklist[cnt]) #Calculate distance between beats in # of samples
    ms_dist = ((RR_interval / fs) * 1000.0) #Convert sample distances to ms distances
    RR_list.append(ms_dist) #Append to list
    cnt += 1
bpm = 60000 / np.mean(RR_list) #60000 ms (1 minute) / average R-R interval of signal
print ("Average Heart Beat is: %.01f" %bpm) #Round off to 1 decimal and print
if (bpm>100):
    print (" Heart Condition : TACHYCARDIA ")
elif(bpm<60):
    print(" Heart Condition : BRADYCARDIA ")
else:
    print (" Heart Condition : NORMAL ")
    


CycleTime = 60000/bpm    # Time duration of a single cycle (60*1000)/bpm in ms 
STStartTime = CycleTime*0.055   # CycleTime*(110/1000)/2 = 0.055 
STStartTimeSample = round (STStartTime*0.250)  # converting ms to sample value
STDuration = CycleTime*0.120  # Duration of ST 
STDurationSample = round (STDuration*0.250) # converting ms to sample value

S_Point=[]
count=0
while (count <= (len(peaklist)-1)):
    S = (peaklist[count] + STStartTimeSample)
    S_Point.append(S)
    count +=1
    print (S_Point)
y1beat = [dataset.hart[x1] for x1 in S_Point] #Get the y-value of S points for plotting purposes


T_Point=[]
count=0
while (count <= (len(S_Point)-1)):
    T = (S_Point[count] + STDurationSample)
    T_Point.append(T)
    count +=1
    print (T_Point)
y2beat = [dataset.hart[x2] for x2 in T_Point] #Get the y-value of T points for plotting purposes


ST = []
ST_interval = np.subtract(y2beat,y1beat)
ST.append(ST_interval)
print(ST)

slope=[]
slope_is =np.divide(ST,STDurationSample)
slope.append(slope_is)
print(slope)

avg_slope =np.mean(slope)
print ("Average Slope is: %.01f" %avg_slope)

if (avg_slope>0.35):
    print (" Heart Condition : MYOCARDIAL ISCHEMIA ")
else:
    print (" Heart Condition : NORMAL ")



plt.title("Detected R peaks in signal")
plt.xlim(0,1000)
plt.xlabel('Time')
plt.ylabel('Amplitude of Signal')
plt.plot(dataset.hart, alpha=0.5, color='black', label="raw signal") #Plot semi-transparent HR
#plt.plot(mov_avg, color ='red', label="moving average") #Plot moving average
plt.scatter(peaklist, ybeat, color='purple', label="average: %.1f BPM" %bpm) #Plot detected peaks
plt.scatter(S_Point, y1beat, color='red',label="ST Segment Range")
plt.scatter(T_Point, y2beat, color='red')
plt.legend(loc=4, framealpha=0.6)
plt.show()
