import os
import shutil

# def get_folders(path):
#     folders = []
#     for item in os.listdir(path):
#         item_path = os.path.join(path, item)
#         if os.path.isdir(item_path):
#             folders.append(item)
#     return folders

# # Example usage
# folder_path = r"C:\Users\22797\Downloads\pde_cons_opt_code"
# folders = get_folders(folder_path)

metrics = ['error', \
          'Total_eq_cons_Loss', \
            'Total_Loss', \
                'Total_l_k_Loss', \
                    'True_sol', \
                        'True_sol_heatmap', \
                            'u_theta', \
                                'u_theta_heatmap']


# def check_path(folder_path):
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
# for metric in metrics:
#     check_path(os.path.join(r"C:\Users\22797\Downloads", metric))



# for metric in metrics:
#     destination_path = os.path.join(r"C:\Users\22797\Downloads", metric)
#     for experiment in folders:
#         dir = os.path.join(folder_path, experiment, "pics", "sin")
#         each_type = os.path.join(dir, metric)
#         for item in os.listdir(each_type):
#             item_path = os.path.join(each_type, item)
#             shutil.move(item_path, destination_path)

import pandas as pd
for item in os.listdir(os.path.join(r"C:\Users\22797\Downloads", "error")):
    print(pd.read_csv(os.path.join(r"C:\Users\22797\Downloads", "error", item)))




