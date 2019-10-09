import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


#file_name = "BanditTenArmedRandomRandom1000_latest"
file_name = "BanditTwoArmedHighHighFixed1000"
db = pickle.load(open(file_name + ".p","rb"))
x = db['x']
y = db['y']
e = db['e']

plt.errorbar(x, y, e)
plt.show()
