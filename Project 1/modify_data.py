import pickle
import numpy as np

with open('network.pickle','rb') as f:
    data = pickle.load(f)

x = data['x']
n = len(x)
x_mod = []
y_mod = []
for i in range(n):
    if x[i][32] == 1:
        p = x[i][:28]
        q = x[i][33:]
        r = np.hstack((p,q))
        x_mod.append(r)
        y_mod.append(data['y'][i])

print len(x_mod)
print len(x_mod[0])
data['x'] = x_mod
data['y'] = y_mod
with open('network_wf5.pickle','wb') as f:
    pickle.dump(data,f)

