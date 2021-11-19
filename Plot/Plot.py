import matplotlib
import notebook as notebook
import numpy
import matplotlib as plt
from scipy import signal, misc
import time

ecg = misc.electrocardiogram()
N = 4000
fs = 360
fn = fs/2
ecg = ecg[0:N]


plt.rcParams['animation.html'] = 'jshtlm'

fig = plt.figure()
ax = fig.add_subplot(111)
fig.show()

i = 0
x, y, new = [], [], numpy.array([])
memory = numpy.array([0]*8)
fc_powerline = numpy.array([49, 51])
wc_powerline = fc_powerline / fn
b, a = sugnal.butter(4, wc_powerline, btype='bandstop', analog=False, output='ba', fs=None)

while i < N:
    x.append(i)
    new = numpy.append(new, [ecg[i]])
    if len(new) == 10:
        new_filtered, memory = sgnal.lfilter(b, a, new, axis=-1, zi=memory)
        y = numpy.array(list(y) + list(new_filtered))
        new = []
        new_filtered = []
        ax.plot(x, y, color='b')
        fig.canvas.draw()
        time.sleep(0.1)
        ax.set_xlim(left=max(0, i-50), right=i+50)
    i += 1

plt.close()



