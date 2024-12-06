import numpy as np
import matplotlib.pyplot as plt

# Monte Carlo Simulation for Load Capacity
def monte_carlo_simulation(material_strength_mean, material_strength_std, num_simulations, load_mean, load_std):
    """
    Simulate the probability of bridge failure due to random material strength and load conditions.
    
    Parameters:
    - material_strength_mean: Average material strength (kg)
    - material_strength_std: Standard deviation of material strength
    - num_simulations: Number of Monte Carlo iterations
    - load_mean: Average applied load (kg)
    - load_std: Standard deviation of applied load
    """
    np.random.seed(42)  # For reproducibility

    # Generate random material strengths and loads
    material_strengths = np.random.normal(material_strength_mean, material_strength_std, num_simulations)
    loads = np.random.normal(load_mean, load_std, num_simulations)

    # Calculate failures (Load > Material Strength)
    failures = loads > material_strengths
    failure_rate = np.sum(failures) / num_simulations

    print(f"Failure Rate: {failure_rate * 100:.2f}%")
    
    # Plot histogram of material strengths and loads
    plt.figure(figsize=(10, 6))
    plt.hist(material_strengths, bins=30, alpha=0.7, label="Material Strength (kg)", color="blue")
    plt.hist(loads, bins=30, alpha=0.7, label="Applied Load (kg)", color="red")
    plt.axvline(material_strength_mean, color='blue', linestyle='--', label="Mean Material Strength")
    plt.axvline(load_mean, color='red', linestyle='--', label="Mean Load")
    plt.title("Monte Carlo Simulation: Material Strength vs Load")
    plt.xlabel("Load/Strength (kg)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.show()

    return failures


# Parameters
material_strength_mean = 100  # Average material strength (kg)
material_strength_std = 10    # Standard deviation of material strength
load_mean = 90                # Average applied load (kg)
load_std = 15                 # Standard deviation of applied load
num_simulations = 10000       # Number of Monte Carlo iterations

# Run simulation
failures = monte_carlo_simulation(material_strength_mean, material_strength_std, num_simulations, load_mean, load_std)
