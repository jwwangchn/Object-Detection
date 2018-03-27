import os
import pickle
import matplotlib.pyplot as plt

def draw_pr(algorithms, save_file, colors):
    fig = plt.figure()
    
    for algorithm, file_name in algorithms:
        color_algorithm = colors[algorithm]
        f = open(file_name)
        info = pickle.load(f)
        x = info['rec']
        y = info['prec']
        print(algorithm, info['ap'])
        plt.plot(x, y, label=algorithm + ' ' + '(' + 'AP = ' + str(round(info['ap'], 3)) + ')', color=color_algorithm)
    
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    fig.savefig(save_file, bbox_inches='tight')
    plt.show()
