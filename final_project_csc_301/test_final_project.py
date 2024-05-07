import numpy as np
from PIL import Image


def convert_image_matrix(image):
    pix = image.load()
    x, y = np.shape(image)[0], np.shape(image)[1]
    energy_rbg = np.zeros((y, x, 2), dtype=object)
    energy = 10
    for col in range(image.size[0]):
        for row in range(image.size[1]):
            energy_rbg[col][row][0] = pix[col, row]
            energy_rbg[col][row][1] = energy
    return energy_rbg


def calc_energy(pix, pos):
    pix_shape = np.shape(pix)
    row, col = pos[0], pos[1]
    # if on the border
    if row not in range(1, pix_shape[0] - 1) or col not in range(1, pix_shape[1] - 1):
        return 1000  # set energy to 1000
    else:  # if not on border
        top = pix[row - 1][col][0]
        bottom = pix[row + 1][col][0]
        left = pix[row][col - 1][0]
        right = pix[row][col + 1][0]

        delta_x = np.sum(np.power(np.subtract(left, right), 2))
        delta_y = np.sum(np.power(np.subtract(top, bottom), 2))

        return np.sqrt(delta_x + delta_y)


def map_energy_matrix(matrix):
    for i in range(np.shape(matrix)[0]):
        for j in range(np.shape(matrix)[1]):
            matrix[i][j][1] = calc_energy(matrix, (i, j))


def vertical_path_finder(matrix):
    cumm_path_energy = np.zeros((np.shape(matrix)[0], np.shape(matrix)[1]))
    path = np.zeros((np.shape(matrix)[0], np.shape(matrix)[1]), dtype=object)

    for row in range(np.shape(matrix)[0]):
        for col in range(np.shape(matrix)[1]):
            ## Checking above
            if row == 0:
                cumm_path_energy[row][col] = matrix[row][col][1]
                path[row][col] = None
            else:
                middle = matrix[row - 1][col][1]

                if col - 1 > 0:
                    left = matrix[row - 1][col - 1][1]
                else:
                    left = None

                if col + 1 < np.shape(matrix)[1]-1:
                    right = matrix[row - 1][col + 1][1]
                else:
                    right = None

                if left is None:
                    if middle <= right:
                        cumm_path_energy[row][col] = middle + matrix[row][col][1]
                        path[row][col] = (row-1, col)
                    else:
                        cumm_path_energy[row][col] = right + matrix[row][col][1]
                        path[row][col] = (row - 1, col + 1)
                else:
                    if middle <= left:
                        cumm_path_energy[row][col] = middle + matrix[row][col][1]
                        path[row][col] = (row-1, col)
                    else:
                        cumm_path_energy[row][col] = left + matrix[row][col][1]
                        path[row][col] = (row - 1, col - 1)

    smallest_col=np.argmin(cumm_path_energy[-1])
  
    prev = path[-1][smallest_col]
    full_smallest_path=[(np.shape(matrix)[0]-1,smallest_col),prev]
    while prev!=None:
        row, col=prev
        prev=path[row][col]
        if prev ==None: break
        full_smallest_path.append(prev)
    out = np.empty(len(full_smallest_path), dtype=object)
    out[:] = full_smallest_path
    return np.flip(out)


def remove_and_shift(matrix, positions):

    n, p, dim2 = np.shape(matrix)
    new_matrix = np.zeros((n, p - 1, dim2), dtype=matrix.dtype)
    
    # Convert positions list to a dictionary where keys are rows and values are columns to skip
    skip_dict = {}
    for row, col in positions:
        if row in skip_dict:
            skip_dict[row].append(col)
        else:
            skip_dict[row] = [col]

    # Sort each list of columns to skip for efficient skipping
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

    
if __name__ == '__main__':
    image = Image.open("input.jpg")
    matrix = convert_image_matrix(image)
    iter=20
    
    for i in range(iter):
        print(np.shape(matrix))
        map_energy_matrix(matrix)
        shortest_path=vertical_path_finder(matrix)
        matrix=remove_and_shift(matrix, shortest_path)

