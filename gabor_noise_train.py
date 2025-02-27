import os
from time import time
from datetime import datetime
import gc
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from gabor_noise_renderer import render, render_with_randomizer_color, GaborModel
from utils.ssim import combined_loss


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps') # for MAC
print("device: ", device)

"""
config
"""
data_dir = './'
gt_image_path = os.path.join(data_dir, 'gabor_noise_example.png')
kernel_size = 512
primitive_num = 500
learning_rate = 0.01
epoch_num = 1000
display_interval = 10

"""
initial process
"""
# Get current date and time as string
now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
# Create a directory with the current date and time as its name
directory = os.path.join(data_dir, f"outputs/{now}")
print("output dir: ", directory)
os.makedirs(directory)
rendered_imgs_dir = 'rendered_images'
os.makedirs(os.path.join(directory, rendered_imgs_dir))
compare_imgs_dir = 'compare_images'
os.makedirs(os.path.join(directory, compare_imgs_dir))
json_dir = 'json_files'
os.makedirs(os.path.join(directory, json_dir))

"""
Loading and processing gt image
"""
gt_image = Image.open(gt_image_path)
gt_image = gt_image.resize((kernel_size, kernel_size)) # fit to kernel_size
gt_image = gt_image.convert('RGB')
gt_image_array = np.array(gt_image)
gt_image_array = gt_image_array / 255.0
# width, height, _ = gt_image_array.shape
target_tensor = torch.tensor(gt_image_array, dtype=torch.float32, device=device)
# Saving processed gt image
plt.imsave(os.path.join(directory, "gt_image.png"), target_tensor.detach().cpu().numpy())

# Saving initial weights
with open(os.path.join(directory, 'weights.txt'), mode='a') as f:
    f.write("===" + f"initial settings" + "===\n")
    f.write(f"primitive_num: {primitive_num}\n")
    f.write(f"kernel_size: {kernel_size}\n")
    f.write(f"learning_rate: {learning_rate}\n")
    f.write(f"the number of num_epochs: {epoch_num}\n")
    f.write(f"display_interval: {display_interval}\n")

"""
Setting up primitives
"""
model = GaborModel()
model.create_points(primitive_num=primitive_num, kernel_size=kernel_size, gt_image_array=gt_image_array)

"""
Training
"""
optimizer = model.get_optimizer(learning_rate=learning_rate)
loss_history = []

start_time = time()

# Training loop
for epoch in range(1, epoch_num + 1):
  gc.collect()
  torch.cuda.empty_cache()

  # render
  rendered_image = render(
    kernel_size=kernel_size, 
    model=model, 
    device=device
  )

  # calcurate loss value
  loss = combined_loss(rendered_image, target_tensor, lambda_param=0.2)
  
  optimizer.zero_grad()

  loss.backward()

  optimizer.step()
  
  loss_history.append(loss.item())

  with torch.no_grad():
    # display and save
    if epoch % display_interval == 0:
      print(f"Epoch {epoch}/{epoch_num}, Loss: {loss.item()}, on {model._means.shape[0]} points")
      
      # save output image
      output_img_array = rendered_image.cpu().detach().numpy()
      plt.imsave(os.path.join(directory, rendered_imgs_dir, f'rendered_output_image_{epoch}.png'), output_img_array)


"""
post process
"""
# Showing and saving ellapsed time
ellapsed_time = time() - start_time
print("time(sec): ", ellapsed_time)
with open(os.path.join(directory, 'weights.txt'), mode='a') as f:
    f.write("===\n" + f"ellapsed time(sec): {ellapsed_time}" + "\n")
    f.write("===\n" + f"final loss: {loss_history[-1]}" + "\n===\n")

# Saving loss figure
loss_fname = "loss.png"
plt.plot(range(1, epoch_num + 1), loss_history)
plt.title('Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(directory, loss_fname), bbox_inches='tight')

# Creating randomized color image
with torch.no_grad():
  rendered_image = render_with_randomizer_color(kernel_size=kernel_size, model=model, device=device)
  output_img_array = rendered_image.cpu().detach().numpy()
  plt.imsave(os.path.join(directory, f'randomized_final_image.png'), output_img_array)

# Creating video
rendered_img_paths = []
for i in range(display_interval, epoch_num + 1, display_interval):
    rendered_img_paths.append(os.path.join(directory, rendered_imgs_dir, f"rendered_output_image_{i}.png"))

# Creating a video writer object
rendered_output_writer = imageio.get_writer(os.path.join(directory, 'rendered_output_video.mp4'), fps=2)

# for rendered_img_path, compare_img_paths in zip(rendered_img_paths, compare_img_paths):
for rendered_img_path in rendered_img_paths:
  rendered_img = imageio.imread(rendered_img_path)
  rendered_output_writer.append_data(rendered_img)

rendered_output_writer.close()
