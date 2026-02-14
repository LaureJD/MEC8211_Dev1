# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 18:28:01 2026

@author: wadih
"""

import numpy as np
import matplotlib.pyplot as plt

class Values():
    def __init__(self):
        self.RADIUS = 0.5
        self.NUMBER_OF_ELEMENTS = [5, 10, 25, 50]
        
        self.Deff = 10**-10
        self.S = 2*10**-8
        self.Ce = 20
        
        self.SLO = 2 

class FiniteDifference():
    def __init__(self, number_of_elements=None, radius_of_geometry=None, Deff=None, S=None, Ce=None, SLO=None):
        
        self.R = radius_of_geometry
        self.Ne = number_of_elements
        self.Deff = Deff
        self.S = S
        self.Ce = Ce
        self.SLO = SLO
        
        self.security_check()
        self.Cells, self.Ds = self.cell_vectorisation()
        self.A_matrix, self.b_vector = self.build_system()
        self.solution = self.solve_system()
        
    def security_check(self):
        if not isinstance(self.Ne, int) or not isinstance(self.R, (int, float)):
            raise TypeError("radius_of_geometry and SLO should be numbers while number_of_elements should be an integer")
        if self.Ne <= 0 or self.R <= 0:
            raise ValueError("Both inputs should be positive")
        if self.SLO not in [1, 2]:
            raise ValueError("Not implemented. Available SLO are only 1 or 2.")
        
    def cell_vectorisation(self):
        vector = np.linspace(0, self.R, self.Ne)
        delta_s = self.R / (self.Ne - 1)
        return vector, delta_s
    
    def build_system(self):
        A = np.zeros((self.Ne, self.Ne))
        b = np.zeros(self.Ne)
        
        # Source term
        source = self.S / self.Deff
        
        # Interior nodes (i = 1 to Ne-2)
        if self.SLO == 1:
            for i in range(1, self.Ne - 1):
                r_i = self.Cells[i]
                
                A[i, i-1] = 1/self.Ds**2
                A[i, i] = -2/self.Ds**2 - 1/(r_i * self.Ds)
                A[i, i+1] = 1/self.Ds**2 + 1/(r_i * self.Ds)
                
                b[i] = source
             
        elif self.SLO == 2:
            for i in range(1, self.Ne - 1):
                r_i = self.Cells[i]
                
                A[i, i-1] = 1/self.Ds**2 - 1/(2 * r_i * self.Ds)
                A[i, i] = -2/self.Ds**2
                A[i, i+1] = 1/self.Ds**2 + 1/(2 * r_i * self.Ds)
                
                b[i] = source
            
            
        # Center node (r = 0)
        A[0, 0] = -2/self.Ds**2
        A[0, 1] = 2/self.Ds**2
        b[0] = source
        
        # Boundary node Dirichlet condition
        A[-1, -1] = 1
        b[-1] = self.Ce
        
        return A, b
    
    def solve_system(self):
        return np.linalg.solve(self.A_matrix, self.b_vector)
    
    def analytical_solution(self):
        return (self.S/(4*self.Deff)) * (self.Cells**2 - self.R**2) + self.Ce
    
    def compute_errors(self, analytical):
        numerical = self.solution
        absolute_error = np.abs(numerical - analytical)
        
        L1 = np.sum(absolute_error) / self.Ne
        L2 = np.sqrt(np.sum(absolute_error**2) / self.Ne)
        Linf = np.max(absolute_error)
        
        return L1, L2, Linf

def plot_results(fd_solution, analytical_solution, cells, errors=None):
    plt.figure(figsize=(10, 6))
    plt.plot(cells, fd_solution, 'bo-', label='Numerical FDM', markersize=8)
    plt.plot(cells, analytical_solution, 'r-', label='Analytical', linewidth=2)
    plt.xlabel('Radius r (m)')
    plt.ylabel('Concentration C (mol/m³)')
    plt.title('Steady-State Concentration Profile in Cylindrical Pilier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if errors:
        textstr = f'L₁ = {errors[0]:.2e}\nL₂ = {errors[1]:.2e}\nL∞ = {errors[2]:.2e}'
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def plot_convergence(errors_dict, elements_list):
    """Plot convergence of errors"""
    plt.figure(figsize=(12, 8))
    
    L1 = [errors_dict[N][0] for N in elements_list]
    L2 = [errors_dict[N][1] for N in elements_list]
    Linf = [errors_dict[N][2] for N in elements_list]
    
    dr = [0.5/(N-1) for N in elements_list]

    plt.loglog(dr, L1, 'bo-', label='L₁ error', markersize=8, linewidth=2)
    plt.loglog(dr, L2, 'rs-', label='L₂ error', markersize=8, linewidth=2)
    plt.loglog(dr, Linf, 'g^-', label='L∞ error', markersize=8, linewidth=2)
        
    plt.xlabel('Δr (m)')
    plt.ylabel('Error')
    plt.title('Convergence Study')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()


def main():
    # Initialize parameters
    val = Values()
    
    # Store results for each mesh
    all_solutions = {}
    all_errors = {}
    
    
    for n_elements in val.NUMBER_OF_ELEMENTS:
        print(f"\n{'='*50}")
        print(f"Mesh with {n_elements} nodes")
        print(f"{'='*50}")
        
        # Create finite difference solver
        fd_solver = FiniteDifference(
            number_of_elements=n_elements,
            radius_of_geometry=val.RADIUS,
            Deff=val.Deff,
            S=val.S,
            Ce=val.Ce,
            SLO=val.SLO
        )
        
        # Get numerical solution
        numerical_solution = fd_solver.solution
        
        # Get analytical solution
        analytical_solution = fd_solver.analytical_solution()
        
        # Compute errors
        errors = fd_solver.compute_errors(analytical_solution)
        
        # Store results
        all_solutions[n_elements] = (fd_solver.Cells, numerical_solution, analytical_solution)
        all_errors[n_elements] = errors
        
        # Print results for this mesh
        
        print(f"\nERROR NORMS for {n_elements} nodes:")
        print(f"L1 norm  = {errors[0]:.3e}")
        print(f"L2 norm  = {errors[1]:.3e}")
        print(f"L∞ norm  = {errors[2]:.3e}")
    
    # Plot convergence
    plot_convergence(all_errors, val.NUMBER_OF_ELEMENTS)
    
    # Plot profile for finest mesh
    finest_mesh = val.NUMBER_OF_ELEMENTS[-1]
    cells, num_sol, ana_sol = all_solutions[finest_mesh]
    plot_results(num_sol, ana_sol, cells, all_errors[finest_mesh])
    
    return all_solutions, all_errors

if __name__ == "__main__":
    solutions, errors = main()