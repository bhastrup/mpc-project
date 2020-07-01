
import matplotlib.pyplot as plt

plt.figure(figsize=(16,16))
ctrs = np.zeros((200, 10))
for i in range(0,200):
    mpc.update_market()
    ctrs[i,:] = mpc.ctr
    
plt.plot(ctrs[:,])
plt.show()


np.mean(ctrs[:,])