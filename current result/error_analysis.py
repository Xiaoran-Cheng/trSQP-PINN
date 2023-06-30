import pandas as pd

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

df_list = []
for experiment in ['PINN_experiment', \
                    'l1_Penalty_experiment', \
                    'l2_Penalty_experiment', \
                    'linfinity_Penalty_experiment', \
                    'Cubic_Penalty_experiment']:
    
    for activation in ['sin', \
                            'tanh', \
                            'cos']:
        dirs = os.path.join(parent_dir, \
        "pde_cons_opt_code\\{experiment}\\pics\\{activation}\\error\\{experiment} {activation}_error_df.csv".format(experiment=experiment,\
                                                    activation=activation))

        df = pd.read_csv(dirs)
        df["activation"] = activation
        df["experiment"] = experiment
        df_list.append(df)


df1 = pd.concat(df_list)
df2 = df1[(df1.loc[:,"Beta"] == 30) & (df1.loc[:,"activation"] == "tanh")]
print(df2[df2.absolute_error == df2.absolute_error.min()])
print(df2[df2.l2_relative_error == df2.l2_relative_error.min()])
df2
df1[(df1.loc[:,"Beta"] == 30)]