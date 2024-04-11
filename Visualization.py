# import matplotlib.pyplot as plt
# import os
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# class Visualization:
#     def __init__(self, current_dir) -> None:
#         self.current_dir = current_dir

#     def check_path(self, folder_path):
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)


#     def heatmap(self, data, sol, test, types, experiment, nt, xgrid, color_bar_bounds, figure_type = "None"):
#         color_bar_lower_bound, color_bar_upper_bound = color_bar_bounds
#         x = data[:,0]
#         t = data[:,1]
#         sol = sol.reshape(nt, xgrid)
#         fig = plt.figure(figsize=(10, 6))
#         ax = fig.add_subplot(111)

#         h = ax.imshow(sol.T, interpolation='nearest', cmap='rainbow',
#                     extent=[t.min(), t.max(), x.min(), x.max()],
#                     origin='lower', aspect='auto', vmin=color_bar_lower_bound, vmax=color_bar_upper_bound)
#         # h = ax.imshow(sol.T, interpolation='nearest', cmap='rainbow',
#         #             extent=[t.min(), t.max(), x.min(), x.max()],
#         #             origin='lower', aspect='auto')
#         if figure_type == "True_sol" or figure_type == "None":
#             divider = make_axes_locatable(ax)
#             cax = divider.append_axes("right", size="5%", pad=0.10)
#             cbar = fig.colorbar(h, cax=cax)
#             cbar.ax.tick_params(labelsize=15)

#         ax.set_xlabel('t', size=25, labelpad=1)
#         if "PINN" in figure_type or figure_type == "None" or figure_type == "L2^2":
#             ax.set_ylabel('s', size=25, labelpad=15, rotation='horizontal')
#         ax.legend(
#             loc='upper center',
#             bbox_to_anchor=(0.9, -0.05),
#             ncol=5,
#             frameon=False,
#             prop={'size': 20}
#         )

#         ax.tick_params(labelsize=20)

#         # title_name = "{experiment} {types} {activation} for beta={beta}".format(beta=beta, types=types, experiment=experiment, activation=activation)
#         title_name = "{experiment}".format(types=types, experiment=experiment)
#         # ax.set_title(title_name, fontsize = 15)
#         folder_path = "{current_dir}/result/{test}/{types}/".\
#                     format(types=types, current_dir=self.current_dir, test=test)
#         self.check_path(folder_path)
#         plt.savefig(os.path.join(folder_path, title_name+".jpg"))
#         plt.show()
#         plt.close()


#     def line_graph(self, ls, test, types, experiment, x=None):
#         plt.figure()
#         if x is None:
#             plt.plot(ls)
#         else:
#             plt.plot(x, ls)
#         title_name = "{experiment} {types}".format(types=types, experiment=experiment)
#         # plt.title(title_name)
#         folder_path = "{current_dir}/result/{test}/{types}/".format(types=types, current_dir=self.current_dir, test=test)
#         self.check_path(folder_path)
#         plt.savefig(os.path.join(folder_path, title_name+".jpg"))
#         plt.show()
#         plt.close()


#     # def error_graph(self, df, folder_path, experiment, activation):
#     #     self.check_path(folder_path)
#     #     title_name = "{experiment} {activation}".format( experiment=experiment, activation=activation)
#     #     # df.to_csv(os.path.join(folder_path, title_name+"_error_df.csv"), index = False)
        
#     #     df.plot(x = "Beta", y='absolute_error',  kind='line')
#     #     plt.xlabel('Beta')
#     #     plt.ylabel('absolute_error')
#     #     # plt.title(title_name+' absolute_error over Beta')
#     #     plt.savefig(os.path.join(folder_path, f"{experiment}"+"_absolute_error.jpg"))
#     #     plt.show()
#     #     plt.close() 

#     #     df.plot(x = "Beta", y='l2_relative_error',  kind='line')
#     #     plt.xlabel('Beta')
#     #     plt.ylabel('l2_relative_error')
#     #     # plt.title(title_name+' l2_relative_error over Beta')
#     #     plt.savefig(os.path.join(folder_path, f"{experiment}"+"_l2_relative_error.jpg"))
#     #     plt.show()
#     #     plt.close()





import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Visualization:
    def __init__(self, current_dir) -> None:
        self.current_dir = current_dir

    def check_path(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


    def heatmap(self, data, sol, test, types, experiment, nt, xgrid, color_bar_bounds, figure_type = "None", pretrain=False):
        color_bar_lower_bound, color_bar_upper_bound = color_bar_bounds
        x = data[:,0]
        t = data[:,1]
        sol = sol.reshape(nt, xgrid)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        h = ax.imshow(sol.T, interpolation='nearest', cmap='rainbow',
                    extent=[t.min(), t.max(), x.min(), x.max()],
                    origin='lower', aspect='auto', vmin=color_bar_lower_bound, vmax=color_bar_upper_bound)
        # h = ax.imshow(sol.T, interpolation='nearest', cmap='rainbow',
        #             extent=[t.min(), t.max(), x.min(), x.max()],
        #             origin='lower', aspect='auto')
        if figure_type == "True_sol" or figure_type == "None":
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.10)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=15)

        ax.set_xlabel('t', size=25, labelpad=1)
        if "PINN" in figure_type or figure_type == "None" or figure_type == "L2^2":
            ax.set_ylabel('s', size=25, labelpad=15, rotation='horizontal')
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.9, -0.05),
            ncol=5,
            frameon=False,
            prop={'size': 20}
        )

        ax.tick_params(labelsize=20)

        # title_name = "{experiment} {types} {activation} for beta={beta}".format(beta=beta, types=types, experiment=experiment, activation=activation)
        title_name = "{experiment}".format(types=types, experiment=experiment)
        # ax.set_title(title_name, fontsize = 15)
        if pretrain:
            folder_path = "{current_dir}/pre_result/{test}/{types}/".\
                    format(types=types, current_dir=self.current_dir, test=test)
        else:
            folder_path = "{current_dir}/result/{test}/{types}/".\
                    format(types=types, current_dir=self.current_dir, test=test)
        self.check_path(folder_path)
        plt.savefig(os.path.join(folder_path, title_name+".jpg"))
        plt.show()
        plt.close()


    def line_graph(self, ls, test, types, experiment, x=None, pretrain=False):
        plt.figure()
        if x is None:
            plt.plot(ls)
        else:
            plt.plot(x, ls)
        title_name = "{experiment} {types}".format(types=types, experiment=experiment)
        # plt.title(title_name)
        if pretrain:
            folder_path = "{current_dir}/pre_result/{test}/{types}/".format(types=types, current_dir=self.current_dir, test=test)
        else:
            folder_path = "{current_dir}/result/{test}/{types}/".format(types=types, current_dir=self.current_dir, test=test)
        self.check_path(folder_path)
        plt.savefig(os.path.join(folder_path, title_name+".jpg"))
        plt.show()
        plt.close()


    # def error_graph(self, df, folder_path, experiment, activation):
    #     self.check_path(folder_path)
    #     title_name = "{experiment} {activation}".format( experiment=experiment, activation=activation)
    #     # df.to_csv(os.path.join(folder_path, title_name+"_error_df.csv"), index = False)
        
    #     df.plot(x = "Beta", y='absolute_error',  kind='line')
    #     plt.xlabel('Beta')
    #     plt.ylabel('absolute_error')
    #     # plt.title(title_name+' absolute_error over Beta')
    #     plt.savefig(os.path.join(folder_path, f"{experiment}"+"_absolute_error.jpg"))
    #     plt.show()
    #     plt.close() 

    #     df.plot(x = "Beta", y='l2_relative_error',  kind='line')
    #     plt.xlabel('Beta')
    #     plt.ylabel('l2_relative_error')
    #     # plt.title(title_name+' l2_relative_error over Beta')
    #     plt.savefig(os.path.join(folder_path, f"{experiment}"+"_l2_relative_error.jpg"))
    #     plt.show()
    #     plt.close()

