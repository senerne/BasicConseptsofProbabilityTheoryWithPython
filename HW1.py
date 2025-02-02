# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:48:35 2025

@author: ŞENER
"""

import numpy as np
import pandas as pd

# Parameters
mean = [2, 2]  # Mean vector [μ₁, μ₂]
cov_matrix = [[1, 0.2], [0.2, 2]]  # Covariance matrix with t=2
sample_size = 200

# Generate bivariate normal distribution samples
samples = np.random.multivariate_normal(mean, cov_matrix, sample_size)

# Create a DataFrame for the samples
samples_df = pd.DataFrame(samples, columns=["X1", "X2"])

# Display the samples
print(samples_df)
import matplotlib.pyplot as plt

# Extract X1 and X2 columns from the samples
x1 = samples[:, 0]
x2 = samples[:, 1]

# Create a scatter plot for the bivariate normal distribution samples
plt.figure(figsize=(8, 6))
plt.scatter(x1, x2, alpha=0.7, edgecolors="k")
plt.title("Bivariate Normal Distribution - Scatter Plot", fontsize=14)
plt.xlabel("X1", fontsize=12)
plt.ylabel("X2", fontsize=12)
plt.grid(True)
plt.show()
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Parameters
mean = [2, 2]
cov_matrix = [[1, 0.2], [0.2, 2]]

# Define the bivariate normal distribution
rv = multivariate_normal(mean, cov_matrix)

# Use the existing 200 samples
x1 = samples[:, 0]
x2 = samples[:, 1]

# Calculate the PDF values for the 200 samples
pdf_values = rv.pdf(np.column_stack((x1, x2)))

# Plot the PDF values as a scatter plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(x1, x2, c=pdf_values, cmap="viridis", edgecolor="k")
plt.colorbar(sc, label="PDF Value")
plt.title("PDF of Bivariate Normal Distribution for 200 Samples", fontsize=14)
plt.xlabel("X1", fontsize=12)
plt.ylabel("X2", fontsize=12)
plt.grid(True)
plt.show()
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Parameters for the marginal distributions
mean_x1 = mean[0]  # Mean of X1
var_x1 = cov_matrix[0][0]  # Variance of X1
mean_x2 = mean[1]  # Mean of X2
var_x2 = cov_matrix[1][1]  # Variance of X2

# Define the ranges for X1 and X2
x1_range = np.linspace(min(samples[:, 0]) - 1, max(samples[:, 0]) + 1, 100)
x2_range = np.linspace(min(samples[:, 1]) - 1, max(samples[:, 1]) + 1, 100)

# Define the marginal PDFs for X1 and X2
pdf_x1 = norm.pdf(x1_range, loc=mean_x1, scale=np.sqrt(var_x1))
pdf_x2 = norm.pdf(x2_range, loc=mean_x2, scale=np.sqrt(var_x2))

# Plot the PDFs
plt.figure(figsize=(10, 6))
plt.plot(x1_range, pdf_x1, label="PDF of X1", linewidth=2)
plt.plot(x2_range, pdf_x2, label="PDF of X2", linewidth=2)
plt.title("Marginal PDFs of X1 and X2", fontsize=14)
plt.xlabel("Value", fontsize=12)
plt.ylabel("PDF Value", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Calculate the correlation coefficient between X1 and X2
correlation_coefficient = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]

# Scatter plot for X1 and X2
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.7, edgecolor="k")
plt.title(f"Scatter Plot of X1 and X2\nCorrelation Coefficient: {correlation_coefficient:.2f}", fontsize=14)
plt.xlabel("X1", fontsize=12)
plt.ylabel("X2", fontsize=12)
plt.grid(True)
plt.show()

# Print the correlation coefficient
print(f"Correlation Coefficient between X1 and X2: {correlation_coefficient:.2f}")
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Define the new random variable Z as Z = X + Y
z = samples[:, 0] + samples[:, 1]  # Z = X1 + X2

# Calculate the expected value (mean) of Z
e_z = np.mean(z)

# Calculate the variance of Z
var_z = np.var(z)

# Create a histogram for Z and overlay the PDF
plt.figure(figsize=(10, 6))

# Plot histogram of Z
plt.hist(z, bins=20, density=True, alpha=0.6, color='b', edgecolor='k', label="Histogram of Z")

# Plot the theoretical PDF of Z
z_range = np.linspace(min(z), max(z), 100)
mean_z = e_z  # Theoretical mean of Z
std_z = np.sqrt(var_z)  # Theoretical standard deviation of Z
pdf_z = norm.pdf(z_range, loc=mean_z, scale=std_z)
plt.plot(z_range, pdf_z, 'r-', label="PDF of Z")

# Add title and labels
plt.title("Distribution of Z = X + Y", fontsize=14)
plt.xlabel("Value of Z", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Print results for E[Z] and Var[Z]
print(f"E[Z] (Expected Value): {e_z:.2f}")
print(f"Var[Z] (Variance): {var_z:.2f}")
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Parameters for Z
mean_z = e_z  # Expected value of Z (calculated earlier)
std_z = np.sqrt(var_z)  # Standard deviation of Z (calculated earlier)

# Define the range for Z
z_range = np.linspace(min(z), max(z), 100)

# Calculate the PDF of Z
pdf_z = norm.pdf(z_range, loc=mean_z, scale=std_z)

# Plot the PDF of Z
plt.figure(figsize=(10, 6))
plt.plot(z_range, pdf_z, 'r-', label="PDF of Z")
plt.title("PDF of Z = X + Y", fontsize=14)
plt.xlabel("Value of Z", fontsize=12)
plt.ylabel("PDF Value", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
# Calculate the correlation coefficient between X1 and Z
correlation_x1_z = np.corrcoef(samples[:, 0], z)[0, 1]

# Calculate the covariance between X1 and Z
covariance_x1_z = np.cov(samples[:, 0], z)[0, 1]

# Scatter plot for X1 and Z
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], z, alpha=0.7, edgecolor="k")
plt.title(f"Scatter Plot of X1 and Z\nCorrelation: {correlation_x1_z:.2f}, Covariance: {covariance_x1_z:.2f}", fontsize=14)
plt.xlabel("X1", fontsize=12)
plt.ylabel("Z = X + Y", fontsize=12)
plt.grid(True)
plt.show()

# Print results
print(f"Correlation: {correlation_x1_z:.2f}")
print(f"Covariance: {covariance_x1_z:.2f}")
# Calculate the correlation coefficient between X1 and Z
correlation_x1_z = np.corrcoef(samples[:, 0], z)[0, 1]

# Calculate the covariance between X1 and Z
covariance_x1_z = np.cov(samples[:, 0], z)[0, 1]

# Scatter plot for X1 and Z
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], z, alpha=0.7, edgecolor="k")
plt.title(f"Scatter Plot of X1 and Z\nCorrelation: {correlation_x1_z:.2f}, Covariance: {covariance_x1_z:.2f}", fontsize=14)
plt.xlabel("X1", fontsize=12)
plt.ylabel("Z = X + Y", fontsize=12)
plt.grid(True)
plt.show()

# Print results
print(f"Correlation: {correlation_x1_z:.2f}")
print(f"Covariance: {covariance_x1_z:.2f}")
# Calculate the correlation coefficient between X1 and Z
correlation_x1_z = np.corrcoef(samples[:, 0], z)[0, 1]

# Calculate the covariance between X1 and Z
covariance_x1_z = np.cov(samples[:, 0], z)[0, 1]

# Scatter plot for X1 and Z
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], z, alpha=0.7, edgecolor="k")
plt.title(f"Scatter Plot of X1 and Z\nCorrelation: {correlation_x1_z:.2f}, Covariance: {covariance_x1_z:.2f}", fontsize=14)
plt.xlabel("X1", fontsize=12)
plt.ylabel("Z = X + Y", fontsize=12)
plt.grid(True)
plt.show()

# Print results
print(f"Correlation between X1 and Z: {correlation_x1_z:.2f}")
print(f"Covariance between X1 and Z: {covariance_x1_z:.2f}")
