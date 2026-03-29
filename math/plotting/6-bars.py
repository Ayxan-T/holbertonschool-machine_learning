#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(8, 6.4))

    fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
    people_names = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    width = 0.5

    # Calculate cumulative sums
    fruit = np.cumsum(fruit, axis=0)

    x = np.arange(3)
    for i in range(3, -1, -1):
        plt.bar(x, fruit[i], width, label=f'{fruit_names[i]}', color=colors[i])
    plt.xticks(x, people_names)
    plt.legend()
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10))
    plt.title("Number of Fruit per Person")

    plt.show()