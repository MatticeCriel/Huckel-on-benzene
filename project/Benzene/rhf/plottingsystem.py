import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf
def taking_carbon_coordinates(coordinates):
    carbon_lines = [line.split()[1:] for line in coordinates.split('\n') if line.startswith('  C') or line.startswith('  N') or line.startswith('  O')]
    carbon_coordinates = np.array(carbon_lines, dtype=float)
    return carbon_coordinates


def taking_x_and_y_coordinate(carbon_coordinates):
    # Verwissel de eerste en laatste kolom
    if carbon_coordinates[0][0].round(2) == 0:
        carbon_coordinates[:, [0, -1]] = carbon_coordinates[:, [-1, 0]]

    # Verwijder rijen waarin de eerste kolom gelijk is aan 0
    carbon_coordinates = carbon_coordinates[:, [0,1]]
    return carbon_coordinates


def taking_indices(molecule_from_gto, orbitals_list):
    pz_indices = [i for i, label in enumerate(molecule_from_gto.ao_labels()) for type_orbital in orbitals_list if type_orbital in label ]
    return pz_indices


def normaliseer(eigenvectors):
    eigenvectors = eigenvectors.T
    for index, eigenvector in enumerate(eigenvectors):
        eigenvectors[index] = eigenvector * (np.dot(eigenvector, eigenvector))**(-1/2)
    return eigenvectors.T


def making_D(coefficients_AO_MO, number_electrons):
    r, k = coefficients_AO_MO.shape
    D = np.zeros((r,r))
    elec_to_place = number_electrons
    i = 0
    while elec_to_place >0:
        if elec_to_place > 1:
            elec_in_orbital = 2
        elif elec_to_place == 1:
            elec_in_orbital = 1
        else:
            elec_in_orbital = 0

        D += elec_in_orbital* ((coefficients_AO_MO[:,i]).reshape(-1,1) @ (coefficients_AO_MO[:,i]).reshape(1,-1))
        elec_to_place -= elec_in_orbital
        i+=1
    return D


def plot_MOs(eigenvalues, eigenvectors, coordinates, extra_bounds, fig_size = (4, 4), unit='eV'):
    transposed_eigenvectors = eigenvectors.T

    # Bepaal het aantal rijen en kolommen voor subplots
    num_rows = int(np.ceil(len(eigenvalues)/2))
    num_cols = 2
    
 
    # Plot de moleculaire orbitalen
    fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size, dpi=400)
    
    for i in range(len(eigenvalues)):
        row = i // num_cols
        col = i % num_cols
        if num_rows >1:
            ax = axs[row, col]
        else:
            ax = axs[col]
        ax.set_aspect('equal')

        line_x = coordinates[:, 0]
        line_y = coordinates[:, 1]
        
        ax.plot(line_x, line_y, linestyle='-', color='grey')
        for x, y in extra_bounds:
            ax.plot([line_x[x], line_x[y]], [line_y[x], line_y[y]], linestyle='-', color='grey')
        
        # Plot de atoomcoëfficiënten als gekleurde bollen
        for j in range(len(coordinates)):
            c = transposed_eigenvectors[i, j]

            if c > 0:
                color = 'blue'
            else:
                color = 'red'

            size = abs(c).round(6) * 1500  # Schaal de grootte van de bol op basis van de coëfficiënt
            ax.scatter(line_x[j], line_y[j], s=size, marker='o', color=color, zorder=2)
            
            # Coëfficiënten in de bollen tonen
            fonts = 8
            if abs(c)<0.20:
                fonts = 4
            if c.round(1) != 0:
                ax.text(line_x[j], line_y[j], f'{c:.2f}', ha='center', va='center', fontsize= fonts, color='white', fontweight='bold')

        ax.set_title(f'MO {i + 1}\nEnergy: {eigenvalues[i]:.2f} {unit}')
        
        ax.set_xlim(line_x.min() - 1, line_x.max() + 1)
        ax.set_ylim(line_y.min() - 1, line_y.max() + 1)

        ax.margins(0.3)
        ax.axis('off')
    


    # Zorg ervoor dat de subplots niet overlappen
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Toon de plot
    plt.show()


def plotting_of_D_atoms(number_of_atoms, coordinates, extra_bounds, D, molecule_name, method,fig_size = (4, 4)):
    transposed_eigenvectors = np.diag(D)

    # Maak de plot voor de benzeenring
    plt.figure(figsize=fig_size, dpi=400)
    _x = coordinates[:, 0]
    _y = coordinates[:, 1]
    plt.plot(_x, _y, linestyle='-', color='grey')
    for bound in extra_bounds:
        plt.plot([_x[bound[0]], _x[bound[1]]], [_y[bound[0]], _y[bound[1]]], linestyle='-', color='grey')

    # Plot de atoomcoëfficiënten als gekleurde bollen
    for j in range(number_of_atoms):
        c = transposed_eigenvectors[j]  # Alleen de eerste eigenvector gebruiken

        if c > 0:
            color = 'blue'
        else:
            color = 'red'

        size = abs(c) * 1500  # Schaal de grootte van de bol op basis van de coëfficiënt
        plt.scatter(_x[j], _y[j], s=size, marker='o', color=color, zorder=2)
        # Toon de coëfficiënt in de bol
        fonts = 8
        if abs(c)<0.20:
            fonts = 4
        if c.round(2) != 0:
            plt.text(_x[j], _y[j], f'{c:.2f}', ha='center', va='center', fontsize=fonts, color='white', fontweight='bold')

    plt.title(f'Electron population on atoms: {molecule_name}: {method}')
    
    plt.gca().margins(0.3)
    plt.gca().axis('off')
    plt.axis('equal') 
    

    # Toon de plot
    plt.show()


