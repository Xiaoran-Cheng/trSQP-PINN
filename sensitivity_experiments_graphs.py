''' This python file is for generate sensitivity experiments error plots, as well as varaying PDE coefficient error plots'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
current_dir = os.getcwd().replace("\\", "/")
sys.path.append(parent_dir)

def check_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

check_path(current_dir+"/"+f"Figures/Errors/")
check_path(current_dir+"/"+f"Figures/robustness_experiment/")
check_path(current_dir+"/"+f"Figures/Heatmaps/")


# problems = ['transport','reaction','reaction_diffusion_alpha','reaction_diffusion_tau']
# experiments = ['coef_experiment']


# problems = ['transport','reaction','reaction_diffusion']
# experiments = ['depth_experiment', 'width_experiment', 'N_experiment', "M_experiment"]
problems = ['transport','reaction','reaction_diffusion']
experiments = ['width_experiment']





markers = {
    'penalty': 'o',
    'ALM': '^',
    'SQP': 's'
}

pic_name = {
    'Absolute Error': "absolute_error.png",
    'Relative Error': "relative_error.png",
}

hyparam_name = {
    "beta": r'$\beta$',
    "alpha": r'$\alpha$',
    "tau": r'$\tau$',
    "NN Depth": 'NN Depth',
    "NN Width": 'NN Width',
    "Number of Training Data": 'Number of Training Data',
    "Number of Pretraining Data": 'Number of Pretrain Data'
}



for problem in problems:
    for experiment in experiments:
        if experiment == "coef_experiment":
            data = pd.read_csv(r"C:\Users\22797\OneDrive\桌面\PINN\trSQP-PINN\line_graphs_data\{experiment}\{problem}.csv".replace("\\", '/').format(problem=problem,experiment=experiment))
            print(data.columns[0])
            for error in ['Absolute Error', 'Relative Error']:
                plt.figure(figsize=(9, 5))
                for method in data['Method'].unique():
                    subset = data[data['Method'] == method]
                    hyparam = np.array(subset[data.columns[0]])
                    hyparam = [str(i) for i in hyparam]
                    errors = np.array(subset[error])
                    plt.plot(hyparam, errors, marker=markers[method], label=method, linewidth=2.5)

                plt.yscale('log')
                plt.xlabel(hyparam_name[data.columns[0]], fontsize=33)
                plt.ylabel(error, fontsize=33)
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)
                plt.savefig(current_dir+"/"+f"Figures/Errors/{problem}.{pic_name[error]}",bbox_inches='tight')
                plt.show()
        else:
            fig, axs = plt.subplots(2, 3, figsize=(15, 6))
            legend_handles = []
            legend_labels = []
            for i, problem in enumerate(problems):
                for j, error in enumerate(['Absolute Error', 'Relative Error']):
                    ax = axs[j, i]
                    data = pd.read_csv(r"C:\Users\22797\OneDrive\桌面\PINN\trSQP-PINN\line_graphs_data\{experiment}\{problem}.csv".replace("\\", '/').format(problem=problem, experiment=experiment))

                    for method in data['Method'].unique():
                        subset = data[data['Method'] == method]
                        hyparam = np.array(subset[data.columns[0]])
                        hyparam = [str(i) for i in hyparam]
                        errors = np.array(subset[error])
                        line, = ax.plot(hyparam, errors, marker=markers[method], label=method, linewidth=2.5)
                        if method not in legend_labels:
                            legend_handles.append(line)
                            legend_labels.append(method)
                        

                    ax.set_yscale('log')
                    if (j,i) in [(1,0),(1,1),(1,2)]:
                        ax.set_xlabel(hyparam_name[data.columns[0]], fontsize=18, labelpad = 10)
                    if (j,i) in [(0,0),(1,0)]:
                        ax.set_ylabel(error, fontsize=18)
                    if (j,i) in [(0,0),(0,1),(0,2)]:
                        ax.set_title(problem.capitalize().replace("_", "-"), fontsize=24, fontweight="bold", pad=20)
                    ax.tick_params(axis='x', labelsize=15)
                    ax.tick_params(axis='y', labelsize=15)

            fig.legend(handles=legend_handles, labels=["Penalty method", "Augmented Lagrangian method", "trSQP-PINN"],
                    loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=18, ncol=3, frameon=False)
            file_name = hyparam_name[data.columns[0]].replace(" ", "_")
            plt.savefig(current_dir+"/Figures/robustness_experiment/"+f"{file_name}.png", bbox_inches='tight')
            plt.show()