import random
import termtables
import tty, sys, termios
import os


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
        self.parent: Cell = None

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

        
    def __repr__(self):
        return "'" + self.value + "'"
    
    def __str__(self):
        return self.value


class Board:
  
    def __init__(self, dimensions: int, min_walls: int = 0, max_walls: int = 10):
        assert min_walls <= max_walls
        assert max_walls < dimensions ** 2 - 2
        
        self.dimensions: int = dimensions
        self.min_walls: int = min_walls
        self.max_walls: int = max_walls
        self.path: list[tuple[int, int]] = []

        _numbered_board: list[list[int]] = list(chunks(list(range(self.dimensions ** 2)), self.dimensions))
        _numbered_board.reverse()
        self.numbered_board: list[list[int]] = _numbered_board

        self.generate_board()

    def generate_base_board(self):
        board = []
        for column_i in range(self.dimensions):
            row = []
            for row_i in range(self.dimensions):
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
        

        self.start_cell: Cell = self.get_cell_by_coordinates(start)
        self.start_cell.value = 's'
        self.end_cell: Cell = self.get_cell_by_coordinates(end)
        self.end_cell.value = 'e'
        
        for wall in walls:
            wall_cell: Cell = self.get_cell_by_coordinates(wall)
            wall_cell.value = 'x'
        
    def generate_board(self):
        self.generate_base_board()
        self.fill_board()
        return self.board

    
    def get_cell_by_coordinates(self, coordinates: tuple[int, int]) -> Cell:
        for column in self.board:
            for cell in column:
                if cell.coordinates == coordinates:
                    return cell
    
    def print(self):
        termtables.print(self.board)

    
    def astar(self):
        """Returns a list of tuples as a path from the given start to the given end in the given maze"""

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(self.start_cell)
        finalpath = []

        # Loop until you find the end
        while len(open_list) > 0:
            #time.sleep(1)
            #print(open_list)
            # Get the current node
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            finalpath.append(current_node.coordinates)

            if current_node.coordinates == self.end_cell.coordinates:
                self.path = finalpath
                return self.path

                # TODO: Fix reverse path traversal
                path = []
                current = current_node
                while current is not None:
                    path.append(current.coordinates)
                    current = current.parent
                
                self.path = path[::-1] # Return reversed path
                return self.path
            
            
            # Loop through children
            for child in self.get_node_children(current_node):

                # Child is on the closed list
                if child in closed_list:
                    continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.coordinates[0] - self.end_cell.coordinates[0]) ** 2) + ((child.coordinates[1] - self.end_cell.coordinates[1]) ** 2)
                #child.h = mhd(child.coordinates, self.end_cell.coordinates)
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                open_list.append(child)

    def get_node_children(self, current_node):
        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.coordinates[0] + new_position[0], current_node.coordinates[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (self.dimensions - 1) or node_position[0] < 0 or node_position[1] > (self.dimensions - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if self.get_cell_by_coordinates(node_position).value not in [' ', 'e']:
                continue

            new_node = self.get_cell_by_coordinates(node_position)
            new_node.parent = current_node

            children.append(new_node)
        return children

    
    def reconstruct_path(self, last, reversePath=False):
        def _gen():
            current = last
            while current:
                yield current.coordinates
                #print(current.parent.coordinates)
                current = current.parent
        if reversePath:
            return _gen()
        else:
            return list(_gen())[::-1]

    @property
    def label(self):
        label = None
        if self.path:
            move = self.path[1]
            if move == self.start_cell.left:
                label = 'left'
            elif move == self.start_cell.right:
                label = 'right'
            elif move == self.start_cell.down:
                label = 'down'
            elif move == self.start_cell.up:
                label = 'up'
        return label
    
    def save_data(self):
        with open("data_pathfinding.csv", "a") as file:
            file.write(str(self.board) + '@' + self.label + '@1' + '\n')



if __name__ == '__main__':
    for i in range(1000000):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(i)
        board = Board(dimensions=6, min_walls=3, max_walls=10)
        board.print()
        path = board.astar()
        print(path)
        print(board.label)
        if board.label:
            board.save_data()