import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
import csv

if __name__ == "__main__":
    csv_names = ["q_lambda_avg_reward", "sarsa_lambda_avg_reward", "qv_lambda_avg_reward"]
    fig, ax = plt.subplots()
    for name in csv_names:
        for i in [0, 1, 2]:
            with open(name+str(i)+".csv", 'r') as f:
                reader = csv.reader(f)
                the_list = list(reader)[0]
                the_list = [float(num) for num in the_list]
                ax.plot(range(len(the_list)), the_list, label=name+"-"+str(i))
    legend = ax.legend(loc='lower right', shadow=True)
    plt.show()
