import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Visualization:
    def __init__(self, current_dir) -> None:
        self.current_dir = current_dir

    def check_path(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


    def heatmap(self, data, sol, test, types, experiment, nt, xgrid, color_bar_bounds, figure_type="None", pretrain=False, title=None):
        ''' Generate heatmaps for solution and predicted solutions of PDEs '''
        color_bar_lower_bound, color_bar_upper_bound = color_bar_bounds
        x = data[:, 0]
        t = data[:, 1]
        sol = sol.reshape(nt, xgrid)
        fig = plt.figure(figsize=(11.5, 7.5))
        ax = fig.add_subplot(111)

        h = ax.imshow(sol.T, interpolation='nearest', cmap='rainbow',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto', vmin=color_bar_lower_bound, vmax=color_bar_upper_bound)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        
        cbar.ax.tick_params(labelsize=30)

        ax.set_xlabel('t', size=55, labelpad=1, fontweight="bold")
        ax.set_ylabel('x', size=55, labelpad=15, rotation='horizontal', fontweight="bold")
        
        ax.tick_params(labelsize=30)

        ax.set_title(title, fontsize=60, pad=60, fontweight='bold')

        title_name = "{experiment}".format(types=types, experiment=experiment)
        if pretrain:
            folder_path = "{current_dir}/pre_result/{test}/{types}/".format(types=types, current_dir=self.current_dir, test=test)
        else:
            folder_path = "{current_dir}/result/{test}/{types}/".format(types=types, current_dir=self.current_dir, test=test)
        self.check_path(folder_path)
        plt.savefig(os.path.join(folder_path, title_name + ".jpg"), bbox_inches='tight')
        plt.show()
        plt.close()



    def line_graph(self, ls, test, types, experiment, x=None, pretrain=False):
        ''' Generate corresponding line graphs for optimization information '''
        plt.figure()
        if x is None:
            plt.plot(ls)
        else:
            plt.plot(x, ls)
        title_name = "{experiment} {types}".format(types=types, experiment=experiment)
        if pretrain:
            folder_path = "{current_dir}/pre_result/{test}/{types}/".format(types=types, current_dir=self.current_dir, test=test)
        else:
            folder_path = "{current_dir}/result/{test}/{types}/".format(types=types, current_dir=self.current_dir, test=test)
        self.check_path(folder_path)
        plt.savefig(os.path.join(folder_path, title_name+".jpg"))
        plt.show()
        plt.close()
