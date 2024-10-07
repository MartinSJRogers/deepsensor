from deepsensor.model import ConvNP
from deepsensor.data import DataProcessor, TaskLoader
from deepsensor.train import Trainer, set_gpu_default_device
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import lab as B
import torch.optim as optim
from torch import tensor, cuda
from torch.cuda.amp import GradScaler, autocast
import torch
import deepsensor.torch
import os
import time
from memory_profiler import profile
import cProfile
import pstats
import io
start_time = time.time()

# Define the base directory containing the year-based subdirectories
base_dir = '/data/hpcdata/users/jambyr/icenet4/data/amsr2_6250/siconca'

# Define date range of AMSR data

data_range = ("2013-06-01", "2022-06-28") 
train_range = ("2013-06-21", "2022-03-20")
val_range = ("2022-03-20", "2022-06-20")
test_range = ("2022-06-21", "2022-06-28")
"""
data_range = ("2013-06-01", "2016-06-28") 
train_range = ("2013-06-06", "2016-04-20")
val_range = ("2016-04-20", "2016-06-25")
test_range = ("2016-06-26", "2016-06-28")

data_range = ("2013-06-01", "2013-06-28") 
train_range = ("2013-06-21", "2013-06-28")
val_range = ("2013-06-21", "2013-06-25")
test_range = ("2013-06-26", "2013-06-28")
"""
# Every other how many items in train and validation sets
date_subsample_factor = 1
batch_size = 20

# Set device to GPU if available
device = torch.device("cuda" if cuda.is_available() else "cpu")

def profile_function(func, *args, **kwargs):
    """
    Profile a single function call with given arguments.
    """
    pr = cProfile.Profile()
    pr.enable()  # Start profiling
    result = func(*args, **kwargs)  # Call the target function
    pr.disable()  # Stop profiling
    
    # Output the profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('time')  # Sort by 'time' or 'cumulative'
    ps.print_stats()
    print(s.getvalue())  # Print profiling results
    
    return result  # Return the result of the profiled function


@profile
def extract_southern_hemisphere_data(base_dir, date_range):
    """
    Method to extract sourthen hemisphere amsr data from James' directory on the SAN
    
    """
    # Ensure the CPU device is used (if using torch)
    device = torch.device('cpu')
    
    data_list = []  # To store data from all relevant NetCDF files
    time_stamps = [] # To store all time stamps so they can be added as a dimension
    
    # Loop through each year in the date range
    for year in date_range.year.unique():
        # Define the path for the corresponding year subdirectory
        year_dir = os.path.join(base_dir, str(year))
        
        if os.path.exists(year_dir):
            # Loop through each date in the date range
            for date in date_range:
                # Construct the expected file pattern (southern hemisphere)
                file_name_pattern = f"asi-AMSR2-s6250-{date.strftime('%Y%m%d')}-v*.nc"
                
                # Use glob to find the file matching the pattern
                for file in os.listdir(year_dir):
                    if file.startswith(f"asi-AMSR2-s6250-{date.strftime('%Y%m%d')}"):
                        file_path = os.path.join(year_dir, file)
                        
                        # Open the NetCDF file as an xarray Dataset
                        ds = xr.open_dataset(file_path)
                        # Subet to Weddell Sea AOI
                        amsr_subset = ds.isel(x=slice(220, 580), y=slice(755, 1135))
                            
                        # Append the data to the list
                        data_list.append(amsr_subset)
                        time_stamps.append(date)
    missing_dates = date_range[~date_range.isin(time_stamps)]

    # Concatenate data along the 'time' dimension
    combined_data = xr.concat(data_list, dim='time')

    # Assign the 'time' coordinate with the corresponding datetime values
    combined_data = combined_data.assign_coords(time=time_stamps)
    
    # Interpolate any missing dates in the data 
    amsr_ds_interpolated = combined_data.interpolate_na(dim='time', method='linear')
        
    # Drop the unnecessary polar stereographic variable and rename ice variable as sic
    amsr_ds_preprocess = amsr_ds_interpolated.drop_vars('polar_stereographic').rename({'z': 'sic'})
    
    return amsr_ds_preprocess

# Collate AMSR data between relevant dates form James' directory on the SAN
training_data_range = pd.date_range(start=data_range[0], end=data_range[1])
amsr_raw_ds = extract_southern_hemisphere_data(base_dir, training_data_range)

print(' Time to get data from James repo on SAN')
print("--- %s seconds ---" % (time.time() - start_time))
# dataprocessor
data_processor = DataProcessor(x1_name="y", x2_name="x", verbose=True)
amsr_ds = data_processor(amsr_raw_ds)

# Instantiate task loader
task_loader = TaskLoader(
    context = [amsr_ds] *20,
    target = amsr_ds,
    context_delta_t = [ -20, -19, -18, -17, -16,
                       -15, -14, -13, -12, -11, -10, 
                       -9, -8, -7, -6, 
                       -5, -4, -3, -2, -1], 
    target_delta_t = 0,
)

def generate_tasks(dates, progress=True):
    train_tasks = []
    for date in tqdm(dates, disable=not progress):
        task = task_loader(date, context_sampling=["all", "all", "all", "all", "all",
                                                  "all", "all", "all", "all", "all",
                                                  "all", "all", "all", "all", "all",
                                                  "all", "all", "all", "all", "all"],
                           target_sampling="all")
        task.remove_context_nans().remove_target_nans()
        train_tasks.append(task)
    return train_tasks
   
print(' Time to generate tasks')
print("--- %s seconds ---" % (time.time() - start_time))   
# Set up model

def gen_sublists(train_tasks, len_sublists):
    """
    Generate sublists of the tasks so that the model can be trained in batches.
    """
    task_count = len(train_tasks)
    task_entries = list(range(0, task_count))
    tasks_shuffled = np.random.permutation(task_entries)
    sublists = [tasks_shuffled[i:i+len_sublists]
            for i in range(0, len(tasks_shuffled), len_sublists)]
    return sublists

def compute_val_rmse(model, val_tasks):
    errors = []
    target_var_ID = task_loader.target_var_IDs[0][0]  # assume 1st target set and 1D
    for task in val_tasks:
        mean = data_processor.map_array(model.mean(task), target_var_ID, unnorm=True)
        true = data_processor.map_array(task["Y_t"][0], target_var_ID, unnorm=True)
        errors.extend(np.abs(mean - true))
    return np.sqrt(np.mean(np.concatenate(errors) ** 2))

# Train model
def train_model(data_processor, task_loader, train_range, date_subsample_factor, batch_size):
    set_gpu_default_device()
    model = ConvNP(data_processor, task_loader)
    opt = optim.Adam(model.model.parameters(), lr=5e-5)
    val_dates = pd.date_range(val_range[0], val_range[1])[::date_subsample_factor]
    val_tasks = generate_tasks(pd.date_range(val_range[0], val_range[1])[::date_subsample_factor], progress=False)
    epoch_losses = []
    val_rmses = []
    val_rmse_best = np.inf
    trainer = Trainer(model, lr=5e-5)
    for epoch in tqdm(range(5)):
        train_tasks = generate_tasks(pd.date_range(train_range[0], train_range[1])[::date_subsample_factor], progress=False)
        sublists = gen_sublists(train_tasks, batch_size)
        for sublist in sublists:
            # Treat each sublist as a batch, generate per task loss, calculate mean loss and use for model update.
            sublist_losses = []
            for i in sublist:
                task_loss = trainer([train_tasks[i]]) # run each task individually per epoch
                epoch_losses.append(task_loss[0].detach().cpu().numpy())  # Detach scalar value from gradient computation and move to CPU for storage
                sublist_losses.append(task_loss[0].detach().cpu().numpy())
            mean_sublist_loss = np.mean(sublist_losses)
            # Recreate a tensor with the mean value, original dtype, and enable gradient tracking
            mean_sublist_loss_tensor = tensor(mean_sublist_loss, dtype=task_loss[0].dtype, requires_grad=True)            
            mean_sublist_loss_tensor.backward()
            opt.step()
            opt.zero_grad()  # Clear previous gradients at start of each sublist/batch.
        epoch_mean_loss = np.mean(epoch_losses)

        val_rmses.append(compute_val_rmse(model, val_tasks))
        if val_rmses[-1] < val_rmse_best:
            val_rmse_best = val_rmses[-1]
            
    return model


# Train model
def train_model_mixed_precision(data_processor, task_loader, train_range, date_subsample_factor, batch_size):
    set_gpu_default_device()
    model = ConvNP(data_processor, task_loader)
    opt = optim.Adam(model.model.parameters(), lr=5e-5)
    val_dates = pd.date_range(val_range[0], val_range[1])[::date_subsample_factor]
    val_tasks = generate_tasks(pd.date_range(val_range[0], val_range[1])[::date_subsample_factor], progress=False)
    losses = []
    val_rmses = []
    val_rmse_best = np.inf
    trainer = Trainer(model, lr=5e-5)
    scaler = GradScaler() # Create a GradScaler for mixed precision training
    for epoch in tqdm(range(50)):
        epoch_losses = []
        train_tasks = generate_tasks(pd.date_range(train_range[0], train_range[1])[::date_subsample_factor], progress=False)
        sublists = gen_sublists(train_tasks, batch_size)
        for sublist in sublists:
            # Treat each sublist as a batch, generate per task loss, calculate mean loss and use for model update.
            sublist_losses = []
            for i in sublist:
                with autocast(): # Use autocast for mixed precision
                    task_loss = trainer([train_tasks[i]]) # run each task individually per epoch
                sublist_losses.append(task_loss[0].detach()) # Detach scalar value from gradient computation
            mean_sublist_loss_tensor = torch.mean(torch.stack(sublist_losses)) 
            # Re-enable gradients for the mean loss tensor
            mean_sublist_loss_tensor_grad = mean_sublist_loss_tensor.requires_grad_()
            #mean_sublist_loss = np.mean(sublist_losses)
            # Recreate a tensor with the mean value, original dtype, and enable gradient tracking
            #mean_sublist_loss_tensor = tensor(mean_sublist_loss, dtype=task_loss[0].dtype, requires_grad=True)  
            
            # Backward pass with scaled loss
            scaler.scale(mean_sublist_loss_tensor_grad).backward()  # Scale the gradients
            scaler.step(opt)  # Update the model parameters
            scaler.update()  # Update the scale for the next iteration
            opt.zero_grad()  # Clear previous gradients at start of each sublist/batch.

            epoch_losses.append(mean_sublist_loss_tensor) # collect all mean losses per sublist
        
        mean_epoch_loss = torch.mean(torch.stack(epoch_losses))
        losses.append(mean_epoch_loss.detach()) # Calculate mean epoch loss as mean of sublist losses
        #print(f"Epoch {epoch + 1} Loss: {mean_epoch_loss.item()}") 
        val_rmses.append(compute_val_rmse(model, val_tasks))
        #print(val_rmses)
        if val_rmses[-1] < val_rmse_best:
            val_rmse_best = val_rmses[-1]
            model.save(f"/data/hpcdata/users/marrog/DeepSensor_code/sea_ice_code/deepsensor_config/model_amsr_pred_gpu_epoch_{epoch}.json")
    for i, loss in enumerate(losses):
        print(f"Epoch {i+1} Loss: {loss.item()}")  # .item() converts the tensor to a Python float        
    for i, loss in enumerate(val_rmses):
        print(f"Epoch {i+1} Val_Loss: {loss.item()}")  # .item() converts the tensor to a Python float        
    

         
    return model



trained_model = train_model_mixed_precision (data_processor, task_loader, train_range, date_subsample_factor, batch_size)
#trained_model = profile_function(train_model, data_processor, task_loader, train_range, date_subsample_factor, batch_size)

#trained_model = profile_function(train_model_mixed_precision, data_processor, task_loader, train_range, date_subsample_factor, batch_size, loss_curve)
print(' Time to train model')
print("--- %s seconds ---" % (time.time() - start_time))

