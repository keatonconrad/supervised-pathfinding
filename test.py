import random
import termtables

num_chunks = 8

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

base_board = list(chunks(list(range(num_chunks**2)), num_chunks))
base_board.reverse()
print(base_board)

def generate_board():
  board = [' '] * num_chunks**2
  nums = list(range(num_chunks**2))
  nums.reverse()
  choices = random.sample(nums, random.randint(0, 19))
  for choice in choices:
    board[choice] = 'x'
  choices = random.sample(nums, 2)
  board[choices[0]] = 's'
  board[choices[1]] = 'e'
  newboard = list(chunks(board, num_chunks))
  return newboard, coord(choices[0]), coord(choices[1])


import tty, sys, termios

filedescriptors = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin)

def coord(num):
    for i, g in enumerate(base_board):
        if num in g:
            row = g.index(num)
            col = i
    return row, col


def dist(n, m):
    return abs(n[0] - m[0]) + abs(n[1] - m[1])


while True:
  import os
  os.system('cls' if os.name == 'nt' else 'clear')
  board, start, end = generate_board()
  termtables.print(board)
  print(start)
  left = (start[0]-1, start[1])
  right = (max(start[0]+1,0), start[1])
  down = (start[0], max(start[1]-1,0))
  up = (start[0], max(start[1]+1,0))
  
  test_board = board
  test_board.reverse()
  try:
    test = test_board[left[1]][left[0]]
    if test == 'x' or left[0] not in list(range(num_chunks-1)) or left[1] not in list(range(num_chunks)):
      raise IndexError
  except IndexError:
    left = (99, 99)
  try:
    test = test_board[right[1]][right[0]]
    if test == 'x' or right[0] not in list(range(num_chunks)) or right[1] not in list(range(num_chunks)):
      raise IndexError
  except IndexError:
    right = (99, 99)
  try:
    test = test_board[down[1]][down[0]]
    if test == 'x' or down[0] not in list(range(num_chunks)) or down[1] not in list(range(num_chunks)):
      raise IndexError
  except IndexError:
    down = (99, 99)
  try:
    test = test_board[up[1]][up[0]]
    if test == 'x' or up[0] not in list(range(num_chunks)) or up[1] not in list(range(num_chunks)):
      raise IndexError
  except IndexError:
    up = (99, 99)
  print(left, right, up, down)


  table = {
    'left': dist(left, end),
    'right': dist(right, end),
    'down': dist(down, end),
    'up': dist(up, end)
  }
  print(table)

  direction = min(table, key=table.get)
  print(direction)

  #x = sys.stdin.read(1)[0]

  print(str(board) + '@' + direction + '@1' + '\n')


  with open("data8x8.csv", "a") as file:
      file.write(str(board) + '@' + direction + '@1' + '\n')