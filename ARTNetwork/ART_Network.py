from collections import defaultdict
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import os
import numpy as np
from art1 import ART
# 5x5 characters & numbers dataset
chars = defaultdict(list)
chars['a'] = [[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [1, 0, 0, 0, 1]]
chars['b'] = [[1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 0]]
chars['c'] = [[1, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
chars['d'] = [[1, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 0]]
chars['e'] = [[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
chars['0'] = [[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]]
chars['1'] = [[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]
chars['2'] = [[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 1, 1, 1, 0]]
chars['3'] = [[1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 1, 1, 1, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
chars['4'] = [[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]]
chars['5'] = [[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
chars['6'] = [[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
chars['7'] = [[1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0]]
chars['8'] = [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
chars['9'] = [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1]]

plt.style.use('ggplot')
colormap = colors.ListedColormap(["white", "black"])

def draw_results(results):
    rows = len(results)
    cols = len(results[0][1]) + 1
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(rows, cols)
    for n_res, res in enumerate(results):
        input = np.array(chars[res[0]]).reshape((5, 5))
        ax = fig.add_subplot(gs[n_res, 0])
        ax.axis('off')
        ax.imshow(input, cmap=colormap)
        if n_res == 0:
            ax.set_title(r'$\mathrm{X}$')
        for j in range(len(res[1])):
            cluster_j = np.reshape(res[1][j, :], (5, 5))
            ax = fig.add_subplot(gs[n_res, j + 1])
            ax.axis('off')
            ax.imshow(cluster_j, cmap=colormap)
            if j == 0 and n_res == 0:
                ax.set_title(r'$\mathbf{z}_{ji}$')
    return fig

if __name__ == '__main__':
    # experiments orders
    orders = [['a', 'b', 'c', 'd', 'e'],
              ['c', 'e', 'a', 'b', 'd'],
              ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
              ['1', '6', '0', '2', '3', '5', '8', '7', '4', '9', '1']]
    # rho configurations
    params = [0.1, 0.3, 0.5, 0.7, 0.9, 0.999]
    # run experiments
    
    Folder = 'figs'
    if not os.path.exists(Folder):
        os.makedirs(Folder)
    
    for n_order, order in enumerate(orders):
        for rho in params:
            nn = ART(25, len(order), rho=rho)
            res = []
            for case in order:
                char_v = np.array(chars[case]).reshape((25, 1))
                out = nn.read_input(char_v)
                # print('\t out = {}'.format(out))
                res.append((case, np.copy(nn.z_ji)))
            # n. of classes learned
            clusters = np.sum([not np.all(nn.z_ji[j]) for j in range(len(nn.z_ji))])
            fig_res = draw_results(res)
            fig_res#.show()
            fig_res.savefig('figs/art_1-order_{}-rho_{:.2f}.png'.format(n_order, rho))
            # print markdown table: cols = {input_order, rho, epochs, clusters}
            print('| {} | {:.3f} | {} | {} |'.format(n_order, rho, nn.epochs, clusters))