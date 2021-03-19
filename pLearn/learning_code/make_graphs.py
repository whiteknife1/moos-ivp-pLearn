import matplotlib.pyplot as plt
from Constants import Constants
Constants=Constants()

with open(Constants.load_model_dir+"model_assessment.log") as f:
    f.readline()
    lines = f.readlines()

performance_metrics=[]

for line in lines:
    metrics = line.split()
    if len(metrics) == 11:
        i = int(metrics[0].split('_')[1][:-1])
        score = int(metrics[3])
        percent_in = int(metrics[7])
        percent_captured = int(metrics[10])
        performance_metrics.append((i, percent_in, score, percent_captured))

fig = plt.figure(1)

plt.subplot(222)
plt.title("Percent Time Inbounds and Untagged")
plt.scatter([x[0] for x in performance_metrics], [x[1] for x in performance_metrics])
plt.plot([x[0] for x in performance_metrics], [x[1] for x in performance_metrics], color = 'blue')

plt.subplot(212)
plt.title("Reward")
plt.scatter([x[0] for x in performance_metrics], [x[2] for x in performance_metrics])
plt.plot([x[0] for x in performance_metrics], [x[2] for x in performance_metrics], color = 'red') 
        
plt.subplot(221)
plt.title("Percent of Successful Captures")
plt.scatter([x[0] for x in performance_metrics], [x[3] for x in performance_metrics])
plt.plot([x[0] for x in performance_metrics], [x[3] for x in performance_metrics], color = 'green') 

fig.savefig(Constants.test_address+"model_graph.png")
#sort the models by reward
performance_metrics.sort(key = lambda x: x[2])
        
#write models out to a file

outfile = open(Constants.test_address+"model_assessment.log", 'w')

#sort top models by percent time inbounds, from highest to lowest
#performance_metrics.sort(key = lambda x: -x[1])
outfile.write("\n----------------Top Performers--------------------\n\n")
for i, percent_in, score, percent_captured in performance_metrics:
    outfile.write("\niteration_"+str(i)+":  avg reward: "+str(score)+" pct time inbounds: "+str(percent_in)+
                    " pct captured: "+str(percent_captured)+"\n")
        
outfile.close()

plt.show()
