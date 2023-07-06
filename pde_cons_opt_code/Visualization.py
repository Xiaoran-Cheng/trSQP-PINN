from jax import numpy as jnp
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

class Visualization:
    def __init__(self, current_dir) -> None:
        self.current_dir = current_dir

    def check_path(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


    def heatmap(self, data, sol, types, experiment, activation, beta, nt, xgrid):
        x = data[:,0]
        t = data[:,1]
        sol = sol.T.reshape(nt, xgrid)
        
        plt.imshow(sol, interpolation='nearest', cmap='rainbow',
                        extent=[t.min(), t.max(), x.min(), x.max()],
                        origin='lower', aspect='auto')
        plt.xlabel('t')
        plt.ylabel('x')
        title_name = "{experiment} {types} {activation} for beta={beta}".format(beta=beta, types=types, experiment=experiment, activation=activation)
        plt.title(title_name)
        folder_path = "{current_dir}/result/beta_{beta}/{types}/".\
                    format(types=types, current_dir=self.current_dir, beta=beta)
        self.check_path(folder_path)
        plt.savefig(os.path.join(folder_path, title_name+".jpg"))
        plt.show()
        plt.close()


    def line_graph(self, ls, types, experiment, activation, beta):
        plt.figure()
        plt.plot(ls)
        title_name = "{experiment} {types} {activation} for beta={beta}".format(beta=beta, types=types, experiment=experiment, activation=activation)
        plt.title(title_name)
        folder_path = "{current_dir}/result/beta_{beta}/{types}/".format(types=types, current_dir=self.current_dir, beta=beta)
        self.check_path(folder_path)
        plt.savefig(os.path.join(folder_path, title_name+".jpg"))
        plt.show()
        plt.close()


    def error_graph(self, df, folder_path, experiment, activation):
        self.check_path(folder_path)
        title_name = "{experiment} {activation}".format( experiment=experiment, activation=activation)
        # df.to_csv(os.path.join(folder_path, title_name+"_error_df.csv"), index = False)
        
        df.plot(x = "Beta", y='absolute_error',  kind='line')
        plt.xlabel('Beta')
        plt.ylabel('absolute_error')
        plt.title(title_name+' absolute_error over Beta')
        plt.savefig(os.path.join(folder_path, f"{experiment}"+"_absolute_error.jpg"))
        plt.show()
        plt.close() 

        df.plot(x = "Beta", y='l2_relative_error',  kind='line')
        plt.xlabel('Beta')
        plt.ylabel('l2_relative_error')
        plt.title(title_name+' l2_relative_error over Beta')
        plt.savefig(os.path.join(folder_path, f"{experiment}"+"_l2_relative_error.jpg"))
        plt.show()
        plt.close()
