import numpy as np
import pandas as pd
from PIL import Image



def convert_image_matrix(image):
    width, height = image.size
    energy_rgb = np.zeros((height, width), dtype=[('color', '3uint64'), ('energy', 'float32')])

    pix = image.load()
    for row in range(height):
        for col in range(width):
            energy_rgb[row][col]['color'] = pix[col, row]  
            energy_rgb[row][col]['energy'] = 10  

    return energy_rgb

def calc_energy(pix, pos):
    row, col = pos
    if row == 0 or row == pix.shape[0] - 1 or col == 0 or col == pix.shape[1] - 1:
        return 1000
    else:
       
        top = pix[row - 1, col]['color']
        bottom = pix[row + 1, col]['color']
        left = pix[row, col - 1]['color']
        right = pix[row, col + 1]['color']

        delta_x = np.sum(np.power(np.subtract(left, right), 2))
        delta_y = np.sum(np.power(np.subtract(top, bottom), 2))
        # Return the magnitude of the gradient
        return np.sqrt(delta_x + delta_y)

def map_energy_matrix(matrix):
    for i in range(np.shape(matrix)[0]):
        for j in range(np.shape(matrix)[1]):
            matrix[i][j]["energy"] = calc_energy(matrix, (i, j))

def vertical_path_finder(matrix):
    rows, cols = matrix.shape[:2]
    cumm_path_energy = np.zeros((rows, cols))
    path = np.zeros((rows, cols), dtype=object)

    # Initialize the first row of cumulative path energy
    for col in range(cols):
        cumm_path_energy[0][col] = matrix[0][col][1]
        path[0][col] = None

    # Compute path energies
    for row in range(1, rows):
        for col in range(cols):
            left = cumm_path_energy[row - 1][col - 1] if col > 0 else float('inf')
            middle = cumm_path_energy[row - 1][col]
            right = cumm_path_energy[row - 1][col + 1] if col < cols - 1 else float('inf')

            # Find the minimum energy and the corresponding column index
            min_energy, offset = min((left, -1), (middle, 0), (right, 1), key=lambda x: x[0])
            
            # Update the cumulative energy and path
            cumm_path_energy[row][col] = matrix[row][col][1] + min_energy
            path[row][col] = (row - 1, col + offset)

    # Retrieve the column of the minimum energy in the last row
    smallest_col = np.argmin(cumm_path_energy[-1])
    current = (rows - 1, smallest_col)
    full_smallest_path = [current]

    # Reconstruct the path from bottom to top
    while path[current[0]][current[1]] is not None:
        current = path[current[0]][current[1]]
        full_smallest_path.append(current)

    return np.array(full_smallest_path[::-1], dtype=object)

def remove_and_shift(matrix, positions):
    n, p = matrix.shape[0], matrix.shape[1]
    # Ensure new_matrix retains the same structured dtype
    new_matrix = np.zeros((n, p - 1), dtype=matrix.dtype)

    # Convert positions list to a dictionary where keys are rows and values are columns to skip
    skip_dict = {}
    for row, col in positions:
        if row in skip_dict:
            skip_dict[row].append(col)
        else:
            skip_dict[row] = [col]

    # Sort the columns to skip for each row for efficient skipping
    for key in skip_dict:
        skip_dict[key].sort()

    # Iterate through each row
    for i in range(n):
        cols_to_skip = skip_dict.get(i, [])
        new_col_index = 0
        skip_index = 0
        for j in range(p):
            if skip_index < len(cols_to_skip) and j == cols_to_skip[skip_index]:
                # If current column is to be skipped, move to next column
                skip_index += 1
                continue
            # Check if we've filled the new matrix's columns
            if new_col_index < p - 1:
                new_matrix[i, new_col_index] = matrix[i, j]
                new_col_index += 1

    return new_matrix

def generate_requiremts_submissions():
    image = Image.open("seam_image_in.jpg")
    matrix = convert_image_matrix(image)
    map_energy_matrix(matrix)
    # print(matrix[1][2]["energy"])
    df_energy_no_pad = pd.DataFrame(matrix[1:-1,1:-1]["energy"])
    df_energy_no_pad.to_csv("energy.csv", index=False, header=False)
    shortest_path=vertical_path_finder(matrix)
    print(shortest_path)
    columns_shortest=np.array([i[1]for i in shortest_path])
    df_cols=pd.DataFrame(columns_shortest)
    df_cols.to_csv("seam1.csv", index=False, header=False)

def matrix_to_jpeg(matrix, filename):
    """
    Converts a matrix of RGB tuples to a JPEG image.

    Args:
    matrix (np.ndarray): A 2D numpy array where each entry is a tuple (r, g, b).
    filename (str): The name of the file to save the JPEG image.
    """

    image = Image.fromarray(np.uint8(matrix), 'RGB')
    image.save(filename, 'JPEG')

if __name__ == '__main__':
    generate_requiremts_submissions()
    image = Image.open("image_test.jpg")
    matrix = convert_image_matrix(image)
    iterations = 100
    for i in range(iterations):
        print(f"--- Iteration {i+1} ---")
        map_energy_matrix(matrix)
        shortest_path = vertical_path_finder(matrix) 
        matrix = remove_and_shift(matrix, shortest_path)
    rgb_data = np.stack([row['color'] for row in matrix])
    matrix_to_jpeg(rgb_data, "final_image.jpg")
    

    
