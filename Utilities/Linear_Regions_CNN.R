# Title     : Linear_Regions_CNN
# Objective : Compute the number of linear regions in CNNs, using Theorem 5, https://arxiv.org/pdf/2006.00978.pdf
# Created by: martin
# Created on: 2020-11-15

library(Rfast)

linear_regions_one_layer_cnn <- function(image_size, filter_size) {


}

linear_regions_cnn <- function(image_sizes) {
  L <- length(image_sizes)
  lower_bound <- 0
  log_R_N_prime <- 5
  d_0 <- image_sizes[[1]][[3]]
  for (i in 2:L) {
    current_image_size <- image_sizes[[i]]
    d_l <- current_image_size[[3]]
    lower_bound <- lower_bound + log(floor(d_l/d_0)) * current_image_size[[1]] * current_image_size[[2]]
  }
  lower_bound <- log_R_N_prime + lower_bound * d_0

  print(paste("CNN lower bound:", lower_bound))
  print(paste("CNN upper bound:", b))
}

linear_regions_fcn <- function(layer_sizes) {
  n_0 <- layer_sizes[[1]]
  n_1 <- layer_sizes[[2]]
  n_2 <- layer_sizes[[3]]
  upper_bound <- 0
  for (j_1 in 0:n_0) {
    for (j_2 in 0:min(n_0, n_1-j_1, n_2)) {
      upper_bound <- upper_bound + Choose(n_1, j_1) * Choose(n_2, j_2)
    }
  }
  lower_bound <- 0
  print(paste("FCN lower bound:", lower_bound))
  print(paste("FCN upper bound:", upper_bound))
}

for (i in 1:8) {
  d_2 <- i
  print(paste("d2:", d_2))
  layer_sizes <- c(4, 6, 2*d_2)
  linear_regions_fcn(layer_sizes)
}