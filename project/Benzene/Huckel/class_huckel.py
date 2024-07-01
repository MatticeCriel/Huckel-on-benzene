import numpy as np
from pyscf import gto, scf
from sympy import symbols, Matrix, simplify, pprint, solve
import matplotlib.pyplot as plt

#deze class heeft de functie om de Mo's in een diagram weer te geven, dit werkt echter enkel met bapaalde parameters.
#Er moet dus nog een afhankelijkheid tussen de parameters en het plotten gevonden worden.
class huckel:
    def __init__(self, number_of_atoms, alpha, beta, configuration='ring'):
        self.number_of_atoms = number_of_atoms
        self.alpha = alpha
        self.beta = beta
        self.configuration = configuration
        self.huckel_matrix = None
        self.huckel_parametric_matrix = 'You did not use the create_parametric_matrix_and_eigenvalues function'
        self.eigval_para = 'You did not use the create_parametric_matrix_and_eigenvalues function'
        self.eigenvalues = 'Try the solve_huckel_matrix method first to obtain this values'
        self.eigenvectors = 'Try the solve_huckel_matrix method to first obtain this eigenvectors'
        self.multiplicities = 'Try the solve_huckel_matrix method to first obtain this multiplicities'
        self.coordinates ='you do not have specified the coordinates yet (can make the molecule in iqmol and copy the coordinates)'
    def __repr__(self):
        return str(self.huckel_matrix)

    
    def __str__(self):
        return f'This is the matrix you made.\n {self.huckel_matrix}'

    
    def create_matrix(self):
        self.huckel_matrix = np.zeros((self.number_of_atoms, self.number_of_atoms))
        for i in range(self.number_of_atoms):
            for j in range(self.number_of_atoms):
                if i == j:
                    self.huckel_matrix[i][j] = self.alpha
                elif abs(i - j) == 1:
                    self.huckel_matrix[i][j] = self.beta
    
        if self.configuration == 'ring':
            self.huckel_matrix[0][self.number_of_atoms-1], self.huckel_matrix[self.number_of_atoms-1][0] = self.beta, self.beta

    
    @staticmethod
    def convert_list_to_latex(expressions):
        latex_list = []

        for expression in expressions:
            terms = str(expression).split()
            latex_representation = ""

            for term in terms:
                if "*" in term:
                    factors = term.split("*")
                    for factor in factors:
                        if factor.isalpha():
                            latex_representation += rf'$\{factor}$'
                        else:
                            latex_representation += rf'{factor}'
                elif term.isalpha():
                    latex_representation += rf'$\{term}$'
                else:
                    latex_representation += term

                latex_representation += " "

            latex_list.append(latex_representation.strip())

        return latex_list
    

    def create_parametric_matrix_and_eigenvalues(self, show = False):
        a, b = symbols('alpha beta')
        self.huckel_parametric_matrix = Matrix([
            [a, b, 0, 0, 0, b],
            [b, a, b, 0, 0, 0],
            [0, b, a, b, 0, 0],
            [0, 0, b, a, b, 0],
            [0, 0, 0, b, a, b],
            [b, 0, 0, 0, b, a]
        ])
        if show:
            pprint(self.huckel_parametric_matrix)

        eigeninfo = self.huckel_parametric_matrix.eigenvects()
        self.eigval_para = np.array([])

        for eigenvalue, multiplicity, eigenvectors in eigeninfo: 
            for _ in eigenvectors:
                self.eigval_para = np.append(self.eigval_para, eigenvalue)
        self.eigval_para = np.flipud(self.eigval_para)
        self.eigval_para_latex = self.convert_list_to_latex(self.eigval_para)


    def solve_huckel_matrix(self, orthogonality = True):
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.huckel_matrix)
        self.eigenvectors = self.eigenvectors.round(6)
        self.eigenvalues = self.eigenvalues.round(6)

        # Sort eigenvalues and corresponding eigenvectors
        sort_indices = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[sort_indices]
        self.eigenvectors = self.eigenvectors[:, sort_indices]
        _, self.multiplicities = np.unique(np.linalg.eigvals(self.huckel_matrix).round(6), return_counts=True)

        if orthogonality: 
            dic_eigenvalues_index = dict()
            for index, eigenvalue in enumerate(self.eigenvalues):
        
                if eigenvalue.round(4) not in dic_eigenvalues_index:
                    dic_eigenvalues_index[eigenvalue.round(4)] = [index]
                    
                else:
                    dic_eigenvalues_index[eigenvalue.round(4)].append(index)
            
            for eigenvalue, indexen in dic_eigenvalues_index.items():
                if len(indexen) > 1:
                    eig_1 = self.eigenvectors[:, indexen[0]]
                    eig_2 = self.eigenvectors[:, indexen[1]]
                    matrix = np.column_stack((eig_1, eig_2))

                    # Voer de QR-decompositie uit
                    q, r = np.linalg.qr(matrix)

                    # De georthogonaliseerde vectoren zijn de kolommen van de Q-matrix
                    orthogonal_v1 = q[:, 0]
                    orthogonal_v2 = q[:, 1]
                    self.eigenvectors[:, indexen[0]] = orthogonal_v1 
                    self.eigenvectors[:, indexen[1]] = orthogonal_v2

        # Dit dient om alle vectoren te normeren nadat ze zijn georthogonaliseerd
        for index, eigenvector in enumerate(self.eigenvectors):
            self.eigenvectors[index] = eigenvector * (np.dot(eigenvector, eigenvector))**(-1/2)


    @staticmethod
    def change_coordinates(coordinates_string):
        # Split the coordinates string into lines and filter out lines that start with 'C'
        carbon_lines = [line for line in coordinates_string.split('\n') if line.strip().startswith('C')]

        # Convert filtered lines directly into a NumPy array
        carbon_coordinates_array = np.array([[float(coord) for coord in line.split()[1:]] for line in carbon_lines])

        # Transpose the array
        transposed_carbon_coordinates_array = carbon_coordinates_array.T

        x_coordinates = transposed_carbon_coordinates_array[0]
        y_coordinates = transposed_carbon_coordinates_array[1]
        z_coordinates = transposed_carbon_coordinates_array[2]


        return x_coordinates, y_coordinates, z_coordinates


    def plotting_of_the_system(self):
        transposed_eigenvectors = self.eigenvectors.T

        # Plot de moleculaire orbitalen
        fig, axs = plt.subplots(1, len(self.eigenvalues), figsize=(15, 3))
        
        for i in range(len(self.eigenvalues)):
            ax = axs[i]
            
            # Plot de benzeenring
            benzene_x = np.cos(2 * np.pi / self.number_of_atoms * np.arange(self.number_of_atoms))
            benzene_y = np.sin(2 * np.pi / self.number_of_atoms * np.arange(self.number_of_atoms))
            ax.plot(benzene_x, benzene_y, linestyle='-', color='grey')
            ax.plot([benzene_x[-1], benzene_x[0]], [benzene_y[-1], benzene_y[0]], linestyle='-', color='grey')
            # Plot de atoomcoëfficiënten als gekleurde bollen
            for j in range(self.number_of_atoms):
                c = transposed_eigenvectors[i, j]

                if c > 0:
                    color = 'blue'
                else:
                    color = 'red'

                size = abs(c) * 300  # Schaal de grootte van de bol op basis van de coëfficiënt
                ax.scatter(benzene_x[j], benzene_y[j], s=size, marker='o', color=color, zorder=2)

            ax.set_title(f'MO {i + 1}\nEnergy: {self.eigenvalues[i]:.2f} eV')

            ax.axis('off')
        # Stel het algehele plot-titel in
        plt.suptitle(f'Molecular Orbitals - Cyclobutadiene - Hückel')

        # Zorg ervoor dat de subplots niet overlappen
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Toon de plot
        plt.show()


    def plotting_of_diagram(self):
        
        # Maak het plot zonder x-as, y-as en kader
        fig, ax = plt.subplots()
        ax.spines['left'].set_visible(2)
        ax.spines['right'].set_linewidth(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])  # Verwijder x-ticklabels

        ax.set_ylim(min(self.eigenvalues) - abs(self.beta), max(self.eigenvalues) + abs(self.beta))
        ax.set_xlim(-1.5, 1.5)

        ax.set_yticks(self.eigenvalues)
        ax.set_yticklabels(self.eigval_para_latex)
        
        counter = 0
        eigval_to = sorted(list(set(self.eigenvalues)))

        for value, multi in zip(eigval_to, self.multiplicities):
            multi = int(multi)

            if multi == 1:
                y_coords = value
                x_coords = 0
                

                benzene_x = np.cos(2 * np.pi / self.number_of_atoms * np.arange(self.number_of_atoms))/abs(2.5*self.alpha) + x_coords
                benzene_y = np.sin(2 * np.pi / self.number_of_atoms * np.arange(self.number_of_atoms))/abs(2.5*self.alpha) + y_coords
                ax.plot(benzene_x, benzene_y, linestyle='-', color='grey')
                ax.plot([benzene_x[-1], benzene_x[0]], [benzene_y[-1], benzene_y[0]], linestyle='-', color='grey')
                # Plot de atoomcoëfficiënten als gekleurde bollen
                for j in range(self.number_of_atoms):
                    c = self.eigenvectors[j, counter]

                    if c > 0:
                        color = 'blue'
                    else:
                        color = 'red'

                    size = abs(c) * 300  # Schaal de grootte van de bol op basis van de coëfficiënt
                    ax.scatter(benzene_x[j], benzene_y[j], s=size, marker='o', color=color, zorder=2)
                counter+=1
                

            else:
                y_coords = [value]* multi
                x_coords = np.linspace(-1/2, 1 + 0.5, num=multi, endpoint=False)
                for x, y in zip(x_coords, y_coords):
                    benzene_x = np.cos(2 * np.pi / self.number_of_atoms * np.arange(self.number_of_atoms))/abs(2.5*self.alpha) + x
                    benzene_y = np.sin(2 * np.pi / self.number_of_atoms * np.arange(self.number_of_atoms))/abs(2.5*self.alpha) + y
                    ax.plot(benzene_x, benzene_y, linestyle='-', color='grey')
                    ax.plot([benzene_x[-1], benzene_x[0]], [benzene_y[-1], benzene_y[0]], linestyle='-', color='grey')
                    # Plot de atoomcoëfficiënten als gekleurde bollen
                    for j in range(self.number_of_atoms):
                        c = self.eigenvectors[j, counter]

                        if c > 0:
                            color = 'blue'
                        else:
                            color = 'red'

                        size = abs(c) * 300  # Schaal de grootte van de bol op basis van de coëfficiënt
                        ax.scatter(benzene_x[j], benzene_y[j], s=size, marker='o', color=color, zorder=2)
                    counter+=1
                    

        ax.set_aspect('equal', adjustable='box')

                
        # Toon het plot (zonder daadwerkelijk een plot te maken)
        plt.show()

    



