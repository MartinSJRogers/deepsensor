# @martin rogers marrog@bas.ac.uk
# SIC workflow for DeepSensor

# Import modules
from deepsensor.data import DataProcessor # I still have this to resolve. 
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import xarray as xr
import numpy as np
import lab as B
from deepsensor.data import TaskLoader
from deepsensor.train import set_gpu_default_device
from deepsensor.model import ConvNP
import deepsensor.torch
import logging
logging.captureWarnings(True)


### User defined vars ###
modis_data_fn = "/data/hpcdata/users/marrog/DeepSensor_code/MODIS_SIC_stack_1kmRes.nc"
amsr_data_fn = "/data/hpcdata/users/marrog/DeepSensor_code/sea_ice_code/SIC_amsr_stack_3031_time_temp5.nc"
train_start_date = "2021-11-23"
train_end_date = "2021-11-25"
test_date = "2021-11-26"
patch_size = (0.6, 0.6)
stride_size = (0.5, 0.5)

### Import data
# Open Netcdf of MODIS-derived SIC with cloud mask
sic_raw_ds = xr.open_dataset(modis_data_fn)
# Open netcdf of AMSR derived SIC and remove land mask
amsr_raw_ds = xr.open_dataset(amsr_data_fn)
amsr_raw_ds = amsr_raw_ds.drop(['land']) # Todo- make land auxillary var

### dataprocessor ###
data_processor = DataProcessor(x1_name="y", x2_name="x", verbose=True)
modis_ds, amsr_ds = data_processor([sic_raw_ds, amsr_raw_ds])

### Instantiate task loader ###
task_loader = TaskLoader(
    context=[amsr_ds, modis_ds]*2,
    target=[modis_ds],
    context_delta_t=[-1, -1, 0, 0],
    target_delta_t=1,
)

### Generate patched tasks ###
# Note when using patching strategy, task comes back as a list of task objects.
train_tasks = []
for date in pd.date_range(train_start_date, train_end_date):
    tasks = task_loader(date, context_sampling=["all", "all", "all", "all"], target_sampling="all",
                       patch_strategy="sliding", patch_size=patch_size, stride=stride_size)
    for task in tasks:
        task.remove_context_nans().remove_target_nans()
        train_tasks.append(task)

### Set up model ###
model = ConvNP(data_processor, task_loader)
opt = optim.Adam(model.model.parameters(), lr=5e-5)

### Train model ###
# Recap: DeepSensor cannot be trained on tasks with a different number of targets.
# To overcome this, the model is trained on each task individually. 
# Model losses are then averaged prior to model update after each sublist. 

# Calculate the number of tasks
task_count = len(train_tasks)
# Produce list [0, task_count]
task_entries = list(range(0, task_count))
# Permute list to assign order tasks applied to model
tasks_shuffled = np.random.permutation(task_entries)
# Set the number of tasks in each sublist
len_sublist = 3
# Split the original list into sublists
sublists = [tasks_shuffled[i:i+len_sublist]
            for i in range(0, len(tasks_shuffled), len_sublist)]
n_epochs = 2

# Todo- add capability to train.py to run this code there? Not a priority for now.
for epoch in tqdm(range(n_epochs)):
    for sublist in sublists:
        task_losses = []
        for i in sublist:
            task_i = train_tasks[i]
            opt.zero_grad()
            task_losses.append(model.loss_fn(task_i, normalise=True))
        mean_batch_loss = B.mean(B.stack(*task_losses))
        mean_batch_loss.backward()
        opt.step()

### Make prediction ###
# Note, data_processor, patch_size and stride_size are additional obligatory inputs predict_patch() compared to predict().
test_task = task_loader(test_date, context_sampling=["all", "all", "all", "all"], target_sampling="all",
                        patch_strategy="sliding", patch_size=patch_size, stride=stride_size)
prediction = model.predict_patch(test_task, X_t=amsr_raw_ds, data_processor = data_processor, patch_size = patch_size, stride_size = stride_size)