import numpy as np
import matplotlib.pyplot as plt


x_0 = np.loadtxt("x_pred_0")
x_1 = np.loadtxt("x_pred_1")


y_pred_0 = np.loadtxt("y_pred_0")
y_pred_1 = np.loadtxt("y_pred_1")


y_act_0 = np.loadtxt("y_act_0")
y_act_1 = np.loadtxt("y_act_1")



plt.plot(x_0, y_pred_0, "-r", lw=1.5,  label="Prediction Rank-0")
plt.plot(x_0, y_pred_1, "--k", lw=1.5, label="Prediction Rank-1")

plt.plot(x_0, y_act_0, "-r", lw=1.5,  label="Actual  Rank-0")
plt.plot(x_0, y_act_1, "--k", lw=1.5, label="Actual Rank-1")

plt.legend()

plt.savefig("func_approax.png", dpi=300)

