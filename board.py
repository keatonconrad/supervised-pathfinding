import random
import termtables
import tty, sys, termios

filedescriptors = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin)

def chunks(lst: list, n: int):
    # Yield successive n-sized chunks from lst.
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def mhd(start: tuple[int, int], end: tuple[int, int]) -> int:
    # Calculates Manhattan distance from two (x, y) coordinates
    if start is None or end is None:
        return None
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


class Cell:
    def __init__(self, coordinates: tuple[int, int], board_dimensions: int):
        self.g: int = 0
        self.h: int = 0
        self.f: int = 0
        self.coordinates: tuple[int, int] = coordinates
        self.value: str = ' '

        self.left = (self.coordinates[0]-1, self.coordinates[1])
        if self.left[0] < 0:
            self.left = None
        self.right = (self.coordinates[0]+1, self.coordinates[1])
        if self.right[0] > board_dimensions - 1:
            self.right = None
        self.down = (self.coordinates[0], self.coordinates[1]-1)
        if self.down[1] < 0:
            self.down = None
        self.up = (self.coordinates[0], self.coordinates[1]+1)
        if self.up[1] > board_dimensions - 1:
            self.up = None

        
    def __str__(self):
        return self.value


class Board:
  
    def __init__(self, dimensions: int, min_walls: int = 0, max_walls: int = 10):
        self.dimensions: int = dimensions
        self.min_walls: int = min_walls
        self.max_walls: int = max_walls

        _numbered_board = list(chunks(list(range(self.dimensions ** 2)), self.dimensions))
        _numbered_board.reverse()
        self.numbered_board = _numbered_board

    def generate_base_board(self):
        board = []
        for row_i in range(self.dimensions):
            row = []
            for column_i in range(self.dimensions):
                cell = Cell(coordinates=(row_i, column_i), board_dimensions=self.dimensions)
                row.append(cell)

            board.insert(0, row)
        
        self.board: list[list[int]] = board

    
    def fill_board(self):
        self.num_walls: int = random.randint(self.min_walls, self.max_walls)

        start = (random.randrange(self.dimensions), random.randrange(self.dimensions))
        end = (random.randrange(self.dimensions), random.randrange(self.dimensions))

        while end == start:
            # If end == start, regenerate end point until end != start
            end = (random.randrange(self.dimensions), random.randrange(self.dimensions))

        excluded: set[tuple[int, int]] = set([start, end])
        walls: list[tuple[int, int]] = []
        
        while len(walls) < self.num_walls:
            new_wall = (random.randrange(self.dimensions), random.randrange(self.dimensions))
            if new_wall in walls or new_wall in excluded:
                continue
            else:
                walls.append(new_wall)
        

        start_cell: Cell = self.get_cell_by_coordinates(start)
        start_cell.value = 's'
        end_cell: Cell = self.get_cell_by_coordinates(end)
        end_cell.value = 'e'
        
        for wall in walls:
            wall_cell: Cell = self.get_cell_by_coordinates(wall)
            wall_cell.value = 'x'
        
    def generate_board(self):
        self.generate_base_board()
        self.fill_board()
        return self.board

    
    def get_cell_by_coordinates(self, coordinates: tuple[int, int]) -> Cell:
        for row in self.board:
            for cell in row:
                if cell.coordinates == coordinates:
                    return cell
    
    def print(self):
        termtables.print(self.board)



if __name__ == '__main__':
    board = Board(dimensions=4, min_walls=0, max_walls=2)
    board.generate_board()
    board.print()