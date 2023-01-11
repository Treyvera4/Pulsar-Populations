#!/usr/bin/env python
# coding: utf-8

# ## Sampling Pulsar Ages

# In this code I sample specific age ranges of a population of pulsars created with the software PsrPopPy. I create a dataframe of pulsar population data in order to plot the relative positions and sizes in an image. The overall goal of the synthetic population is to tell us the inefficient regions of cosmic-ray diffion in the Andromeda Galaxy.

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


ages = np.random.uniform(0E+0,5E+9,500000)


# In[ ]:


ages_cut = ages[(ages >= 1e4) & (ages <= 1e7)]
ages_cut.sort()


# In[ ]:


ages_cut


# In[ ]:


ages_new = ages_cut[:1000]
ages_new


# In[ ]:


np.histogram(ages_new, 10)
plt.hist(ages_new, 10)
plt.show()


# In[ ]:


radius_PWN = (ages_new)**0.3


# In[ ]:


columns = ['Pulsar_ages']


# In[ ]:


index = np.arange(1,1001)


# In[ ]:


pulsars = pd.DataFrame(ages_new, index, columns)


# In[ ]:


pulsars.insert(1, "PWN_radii", radius_PWN, True)


# In[ ]:


pulsars


# In[ ]:


index = np.arange(1,1001)


# In[ ]:


#viewing test_pop

test_pop3 = pd.read_csv("Test_populations/test_pop.csv") 


# In[ ]:


pulsar_data = test_pop3


# In[ ]:


pulsar_data.insert(13, "Pulsar_ages", ages_new, True)
pulsar_data.insert(14, "PWN_radii", radius_PWN, True)


# In[ ]:


pulsar_data


# In[ ]:


pulsar_data.to_csv ('Pulsar_dataframe.csv', sep=',')


# In[ ]:


fig = plt.figure(figsize=(6,4.5), dpi=150)
ax = plt.axes(projection='3d')

c=pulsar_data.PWN_radii #color of points
s = pulsar_data.PWN_radii #size of points
im = ax.scatter(pulsar_data.X, pulsar_data.Y, pulsar_data.Z, c=c, s=s)

plt.xlim([-20, 20])
plt.ylim([-20, 20])

fig.colorbar(im, ax=ax)
#plt.colorbar.set_label('PWN Radii')

plt.show()


# In[ ]:


tau_0 = 1
t_rs = 6
norm2 = 30/((300 - tau_0)**0.3)
norm1 = (norm2*((t_rs - tau_0)**0.3))/((t_rs - tau_0)**1.2)


# In[ ]:


norm2, norm1


# In[ ]:


fig = plt.figure(figsize=(6,4.5), dpi=150)

plt.plot((pulsar_data.Pulsar_ages)/1000, pulsar_data.PWN_radii, color = 'blue')
plt.ticklabel_format(axis="x", style="plain", scilimits=(0,0))
plt.ylabel('PWN radii(pc)')
plt.xlabel('Pulsar ages(kyr)')


# In[ ]:


fig = plt.figure(figsize=(6,4.5), dpi=150)

plt.plot((pulsar_data.Pulsar_ages)/1000, pulsar_data.PWN_radii, color = 'blue')
plt.ticklabel_format(axis="x", style="plain", scilimits=(0,0))
#plt.xlim(0, 100)
#plt.ylim(0, 40)
plt.yscale("log")
plt.xscale("log")
plt.ylabel('PWN radii(pc)')
plt.xlabel('Pulsar ages(kyr)')


# ### Sampling Pulsar Ages (New Population of 10,000)

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


ages = np.random.uniform(1E+3,1E+5,10000)


# In[ ]:


ages.sort()
ages


# In[ ]:


ages_1 = ages[:475]
ages_1


# In[ ]:


ages_2 = ages[475:]
ages_2


# In[ ]:


tau0 = 1000
t0 = 5000 + tau0
pwn = 8.5


# In[ ]:


radius_PWN_1 = (pwn)*(((ages_1) - tau0)/(t0 - tau0))**1.2
radius_PWN_1


# In[ ]:


radius_PWN_2 = (pwn)*(((ages_2) - tau0)/(t0 - tau0))**0.3
radius_PWN_2


# In[ ]:


radius_PWN = np.concatenate((radius_PWN_1, radius_PWN_2))
radius_PWN


# In[ ]:


#np.histogram(ages, 10)
#plt.hist(ages, 10)
#plt.show()


# In[ ]:


#tau0 = 1000
#t0 = 5000 + tau0
#pwn = 8.5

#length = len(ages[:515])
#print(length)

#for i in range(length):
#    ages[i] = ages[i]*2
    
#new_length = len(ages[515:])
#print(new_length) 
    
#for i in range(new_length):
#    ages[i] = ages[i]*4    
#print(ages[:550])


# In[ ]:


#radius_PWN = (ages)**0.3

#tau0 = 1000
#t0 = 5000 + tau0
#pwn = 8.5

#if (ages) > t0:
#     radius_PWN == 8.5*(((ages) - tau0)/(t0 - tau0))^0.3
        
#else:
#    radius_PWN == 8.5*(((ages) - tau0)/(t0 - tau0))^1.2
    
    
#for x in range(ages):
#    print(x)
    
    
#while(ages > 5000+tau0):
#    radius_PWN == 8.5*(((ages) - tau0)/(t0 - tau0))^0.3
#    print('function #1')
#else:
#    radius_PWN == 8.5*(((ages) - tau0)/(t0 - tau0))^1.2
#    print('function #2')


# In[ ]:


columns = ['Pulsar_ages']
index = np.arange(1,10001)


# In[ ]:


pulsars = pd.DataFrame(ages, index, columns)


# In[ ]:


pulsars.insert(1, "PWN_radii", radius_PWN, True)


# In[ ]:


pulsars


# In[ ]:


#viewing new_test_pop

new_test_pop = pd.read_csv("Test_populations/new_test_pop.csv") 


# In[ ]:


pulsar_data = new_test_pop


# In[ ]:


pulsar_data.insert(13, "Pulsar_ages", ages, True)
pulsar_data.insert(14, "PWN_radii", radius_PWN, True)


# In[ ]:


new_index = np.arange(1,10001)
pulsar_data.set_axis([np.arange(1,10001)], axis='index', inplace=True)


# In[ ]:


pulsar_data = pulsar_data.drop(['Period_ms', 'DM', 'Width_ms', 'S1400', 'L1400', 'SPINDEX', 'SNR', 'DTRUE'], axis=1)


# In[ ]:


pulsar_data


# In[ ]:


#pulsar_data.to_csv ('Pulsar_dataframe.csv', sep=',')


# ### Plotting a cropped version of the pulsar dataframe

# In[ ]:


pulsars = pd.read_csv('Official_pulsar_dataframe.csv')
pulsars


# In[ ]:


#new cropped dataframe

new_radius_PWN = radius_PWN[1:10000:50]
new_radius_PWN


# In[ ]:


new_ages = ages[1:10000:50]
new_ages


# In[ ]:


#new cropped coordinate arrays

new_X = pulsar_data.X[1:10000:50]

new_Y = pulsar_data.Y[1:10000:50]

new_Z = pulsar_data.Z[1:10000:50]


# In[ ]:


columns = ['Pulsar_ages']
index = np.arange(1,201)


# In[ ]:


pulsars = pd.DataFrame(new_ages, index, columns)


# In[ ]:


pulsars.insert(1, "PWN_radii", new_radius_PWN, True)
pulsars


# In[ ]:


fig = plt.figure(figsize=(6,4.5), dpi=150)
ax = plt.axes(projection='3d')

#c = blue
#c=pulsars.PWN_radii #color of points
s = 3*pulsars.PWN_radii #size of points
#im = ax.scatter(new_X, new_Y, new_Z, c=c, s=s)
im = ax.scatter(new_X, new_Y, new_Z, s=s)

plt.xlim([-15, 15])
plt.ylim([-15, 15])
ax.set_zlim([-4, 4])

ax.set_title('Swiss Cheese Model (200)')
plt.xlabel('X (pc)')
plt.ylabel('Y (pc)')
ax.set_zlabel('Z (pc)')
#ax.tick_params(axis='z', labelleft=True, left=False)
#ax.zaxis.set_tick_params(labelleft=False)

#ax.tick_params(
 #   bottom=False, top=True,
  #  left=True, right=True)
#ax.tick_params(
 #   labelbottom=False, labeltop=True,
  #  labelleft=True, labelright=True)

#cbar = fig.colorbar(im, ax=ax)
#cbar.ax.set_ylabel('PWN Radii (pc)')

plt.show()
#plt.savefig('Swiss Cheese Model.jpg')


# In[ ]:




