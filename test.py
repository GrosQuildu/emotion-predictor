import pickle
import neurokit as nk
import pandas as pd
import matplotlib.pyplot as plt


def open_file(in_file):
    with open(in_file, 'rb') as fp:
        itemlist = pickle.load(fp)
        return itemlist


data = open_file("sample.dat")
data = pd.DataFrame({"EDA": data})
data.plot()
plt.show()


results = nk.eda_process(data["EDA"], sampling_rate=128)

# Plot standardized data (visually more clear) with SCR peaks markers
nk.plot_events_in_signal(nk.z_score(results["df"]), results["EDA"]["SCR_Peaks_Indexes"])
plt.show()
