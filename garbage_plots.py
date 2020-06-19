plt.figure(figsize=(12,12))
ctrs = np.zeros((200, 10))
for i in range(0,200):
    mpc.update_market()
    ctrs[i,:] = mpc.ctr
    
plt.plot(ctrs[:,])
np.mean(ctrs[:,])