def is_valid_move(grid, row, col, number):
    for i in range(9):
        if (grid[row][i] == number or grid[i][col] == number):    #check the row and column
            return False
        
    leftmost_row = row - row % 3    #gives the leftmost_row in the 3x3 square. row = 5 -> 5 - 2 = 3. 0-indexed, so it's correct
    topmost_column = col - col % 3  #gives topmost column

    for i in range(3):
        for j in range(3):
            if (grid[leftmost_row + i][topmost_column + j] == number):
                return False
            
    return True

#recursive
# def solve(grid, row, column):
#     if column == 9: 
#         if row == 8:
#             return True     #this means it has been solved. if we go PAST the final column and are on the final row, it is solved
        
#         # if row != 8, go to new row and start with new column. we're basically filling it up down a row
#         row += 1
#         column = 0

#     if (grid[row][column] > 0):
#         return solve(grid, row, column + 1)
    
#     for num in range(1, 10):   
#         if is_valid_move(grid, row, column, num):
#             grid[row][column] = num

#             if (solve(grid, row, column + 1)):
#                 return True
            
#         grid[row][column] = 0

#     return False

#let's see if i can write this on my own!

def solve(grid, row, column):
    if (column == 9):
        if (row == 8):
            return True
    
        row += 1
        column = 0

    if (grid[row][column] > 0):
        return solve(grid, row, column + 1)
    
    for i in range(1, 10):
        if (is_valid_move(grid, row, column, i)):
            grid[row][column] = i
            #why not directly return solve here? because need to try with every number 1...9. if 1 fails, reset it to 0 and try it with 2
            if solve(grid, row, column + 1):
                return True
            
            grid[row][column] = 0

    return False
        

grid = [
    [0, 0, 0, 0, 0, 0, 6, 8, 0],
    [0, 0, 0, 0, 7, 3, 0, 0, 9],
    [3, 0, 9, 0, 0, 0, 0, 4, 5],
    [4, 9, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 3, 0, 5, 0, 9, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 3, 6],
    [9, 6, 0, 0, 0, 0, 3, 0, 8],
    [7, 0, 0, 6, 8, 0, 0, 0, 0],
    [0, 2, 8, 0, 0, 0, 0, 0, 0]
]

if (solve(grid, 0, 0)):
    for i in range(9):
        for j in range(9):
            print(grid[i][j], end = " ")
        print()

else:
    print("No solution :(")
