from process_injections import gen_log
from matplotlib import pyplot as plt
import numpy as np
snrs = np.linspace(0,100,1000)

y = gen_log(snrs, 0.212, 0.2699, 0.8731, 0.0, 1.55)
plt.plot(snrs, y)
plt.show()
