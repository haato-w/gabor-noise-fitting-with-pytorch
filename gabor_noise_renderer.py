import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps') # for MAC

class GaborModel:
  def __init__(self):
    self._means = torch.empty(0)
    self._colors = torch.empty(0)
    self._colors2 = torch.empty(0)
    self._frequencies = torch.empty(0)
    self._opacities = torch.empty(0)
    self._scales = torch.empty(0)
    self._rotations = torch.empty(0)
  
  def get_optimizer(self, learning_rate: float):
    optimizer = torch.optim.Adam([
        self._means, 
        self._colors, 
        self._colors2, 
        self._frequencies, 
        self._opacities, 
        self._scales, 
        self._rotations
      ], lr=learning_rate)
    
    return optimizer

  def create_points(self, primitive_num: int, kernel_size: int, gt_image_array: np.ndarray):
    # Generating random center coordinates[-1.0-1.0]
    means = 2 * torch.rand((primitive_num, 2),  dtype=torch.float32, device=device) - 1

    # Fetching the color of the pixels in each coordinates
    points_on_gt_pixel = [list(map(int, kernel_size * ((mean + 1.0) / 2))) for mean in means]
    # print(points_on_gt_pixel)
    color_values_array = [gt_image_array[point[0], point[1]] for point in points_on_gt_pixel]
    color_values_np = np.array(color_values_array)
    colors = torch.tensor(color_values_np, dtype=torch.float32, device=device)
    # colors = 0.1 * torch.ones((primitive_num, 3), dtype=torch.float32, device=device)
    # print("colors: \n", colors)
    colors2 = torch.zeros((primitive_num, 3), dtype=torch.float32, device=device) # black color
    # colors2 = 0.1 * torch.rand((primitive_num, 3), dtype=torch.float32, device=device) # random color
    # print("colors2: \n", colors2)

    frequencies = 0.5 + torch.rand((primitive_num, 2), dtype=torch.float32, device=device)
    opacities = torch.ones(primitive_num, dtype=torch.float32, device=device)
    scales = 0.5 + torch.rand((primitive_num, 2), dtype=torch.float32, device=device)
    rotations = np.pi * torch.rand(primitive_num, dtype=torch.float32, device=device)

    self._means = nn.Parameter(means.requires_grad_(True))
    self._colors = nn.Parameter(colors.requires_grad_(True))
    self._colors2 = nn.Parameter(colors2.requires_grad_(True))
    self._frequencies = nn.Parameter(frequencies.requires_grad_(True))
    self._opacities = nn.Parameter(opacities.requires_grad_(True))
    self._scales = nn.Parameter(scales.requires_grad_(True))
    self._rotations = nn.Parameter(rotations.requires_grad_(True))
  
  def create_simple_points(self):
    means_array = [[1.0, 1.0], [0.5, -0.5]] # ndc
    scales_array = [[1.0, 1.0], [2.0, 1.0]] # ndc
    rotations_array = [0.0, 45.0] # ndc
    frequencies_array = [[2.0, 0.0], [0.0, 1.0]]
    colors_array = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    colors2_array = [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    opacities_array = [1.0, 0.5]

    means = torch.tensor(means_array, device=device)
    frequencies = torch.tensor(frequencies_array, device=device)
    colors = torch.tensor(colors_array, device=device)
    colors2 = torch.tensor(colors2_array, device=device)
    opacities = torch.tensor(opacities_array, device=device)
    scales = torch.tensor(scales_array, device=device)
    rotations = torch.tensor(rotations_array, device=device)

    self._means = nn.Parameter(means.requires_grad_(True))
    self._colors = nn.Parameter(colors.requires_grad_(True))
    self._colors2 = nn.Parameter(colors2.requires_grad_(True))
    self._frequencies = nn.Parameter(frequencies.requires_grad_(True))
    self._opacities = nn.Parameter(opacities.requires_grad_(True))
    self._scales = nn.Parameter(scales.requires_grad_(True))
    self._rotations = nn.Parameter(rotations.requires_grad_(True))


def render_gabor_image(
  kernel_size: int, 
  means: torch.Tensor, 
  frequencies: torch.Tensor, 
  colors: torch.Tensor, 
  colors2: torch.Tensor, 
  opacities: torch.Tensor, 
  scales: torch.Tensor, 
  rotations: torch.Tensor, 
  device
) -> torch.Tensor:
  batch_size = means.shape[0]

  # create object space coordinate grid
  object_space_range = 10.0
  start = torch.tensor([-1.0 * object_space_range], device=device).view(-1, 1)
  end = torch.tensor([1.0 * object_space_range], device=device).view(-1, 1)

  base_linspace = torch.linspace(0, 1, steps=kernel_size, device=device)
  ax_batch = start + (end - start) * base_linspace

  ax_batch_expanded_x = ax_batch.unsqueeze(1).expand(-1, kernel_size, -1).unsqueeze(-1)
  ax_batch_expanded_y = torch.flip(ax_batch, dims=[1]).unsqueeze(-1).expand(-1, -1, kernel_size).unsqueeze(-1)

  xx, yy = ax_batch_expanded_x, ax_batch_expanded_y

  # create gaussian strength grid
  gaussian_strength = torch.exp(-(xx * xx + yy * yy) / 2)

  # reshape frequencies tensor to fit batch grid
  frequencies = frequencies.view(batch_size, 1, 1, 2)
  frequencies_x = frequencies[..., :1]
  frequencies_y = frequencies[..., 1:]
  
  # create harmonic strength
  harmonic1_strength = (1 + torch.cos(2 * np.pi * (xx * frequencies_x + yy * frequencies_y))) / 2
  harmonic2_strength = (1 - torch.cos(2 * np.pi * (xx * frequencies_x + yy * frequencies_y))) / 2

  # reshape colors tensor to fit batch grid
  colors = colors.view(batch_size, 1, 1, 3)
  colors2 = colors2.view(batch_size, 1, 1, 3)

  # create harmonic with color grid for each batch
  harmonic_with_color1 = colors * harmonic1_strength
  harmonic_with_color2 = colors2 * harmonic2_strength

  # reshape opacities tensor to fit batch grid
  opacities = opacities.view(batch_size, 1, 1, 1)

  # create primitive grid
  primitives = opacities * gaussian_strength * (harmonic_with_color1 + harmonic_with_color2)

  # apply scale, rotation and mean to convert to world space
  affine_matrices = torch.zeros((batch_size, 2, 3), dtype=torch.float32, device=device)
  scales_x, scales_y = scales[:, 0], scales[:, 1]
  cos_theta = torch.cos(rotations)
  sin_theta = torch.sin(rotations)
  affine_matrices[:, 0, 0] = scales_x * cos_theta
  affine_matrices[:, 0, 1] = -scales_x * sin_theta
  affine_matrices[:, 1, 0] = scales_y * sin_theta
  affine_matrices[:, 1, 1] = scales_y * cos_theta
  affine_matrices[:, 0, 2] = -means[:, 0] * scales_x * cos_theta + means[:, 1] * (-scales_x * sin_theta)
  affine_matrices[:, 1, 2] = -means[:, 0] * scales_y * sin_theta + means[:, 1] * (scales_y * cos_theta)

  grid = F.affine_grid(affine_matrices, size=(batch_size, 3, kernel_size, kernel_size), align_corners=False)
  primitives = primitives.permute(0, 3, 1, 2).contiguous() # contiguous is needed by grid_sample
  primitives = F.grid_sample(primitives, grid, align_corners=False)
  primitives = primitives.permute(0, 2, 3, 1)
  
  # integrate primitive layers
  image = primitives.sum(dim=0)
  image = torch.clamp(image, min=0.0, max=1.0)

  return image

def render(kernel_size: int, model: GaborModel, device):
  return render_gabor_image(
    kernel_size=kernel_size, 
    means=model._means, 
    frequencies=model._frequencies, 
    colors=model._colors, 
    colors2=model._colors2, 
    opacities=model._opacities, 
    scales=model._scales, 
    rotations=model._rotations, 
    device=device  
  )

def render_with_randomizer_color(kernel_size: int, model: GaborModel, device):
  primitive_num = model._means.shape[0]

  np.random.seed(0)
  colors1 = matplotlib.colormaps['gist_rainbow'](np.random.randint(low=0, high=256, size=primitive_num))[..., :3]
  colors1 = torch.from_numpy(colors1).to(dtype=torch.float32).to(device)
  colors2 = matplotlib.colormaps['gist_rainbow'](np.random.randint(low=0, high=256, size=primitive_num))[..., :3]
  colors2 = torch.from_numpy(colors2).to(dtype=torch.float32).to(device)

  colors1 /= 10
  colors2 /= 10

  return render_gabor_image(
    kernel_size=kernel_size, 
    means=model._means, 
    frequencies=model._frequencies, 
    colors=colors1, 
    colors2=colors2, 
    opacities=model._opacities, 
    scales=model._scales, 
    rotations=model._rotations, 
    device=device  
  )

if __name__ == "__main__":
  kernel_size = 512

  model = GaborModel()
  model.create_simple_points()

  rendered_image = render(
     kernel_size, 
     model, 
     device
  )

  image = rendered_image.clone().detach().cpu().numpy()
  plt.imsave("rendered_image.png", image)
 