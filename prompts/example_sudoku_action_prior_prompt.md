# Sudoku Game Rules
You are helping solve Sudoku puzzles using a tree-based search approach. Sudoku is a puzzle where
you fill a grid with numbers 1 through {grid_size} so that each row, column, and box has no repeated
numbers.
For this {grid_size} × {grid_size} Sudoku grid, the boxes are {box_width} × {box_height} in
size. Each row, column, and box must contain all numbers from 1 to {grid_size} without repetition.
This means:
1. Each row must contain each number from 1 to {grid_size} exactly once
2. Each column must contain each number from 1 to {grid_size} exactly once
3. Each {box_width}×{box_height} box must contain each number from 1 to {grid_size}
exactly once
These constraints create a logical puzzle where placing a number in a cell immediately restricts what
numbers can be placed in other cells in the same row, column, and box.
Board Structure:
• The Sudoku board is a {grid_size} × {grid_size} grid divided into {box_width} ×
{box_height} boxes
• Rowsare numbered 0 to {grid_size_minus_one} from top to bottom
• Columns are numbered 0 to {grid_size_minus_one} from left to right
• Each cell is identified by its (row, column) coordinates
• Empty cells appear as periods (.) in the board representation
• Board state is represented as a nested list where board[row][column] gives the value at
that position
When solving a Sudoku puzzle, we explore different possible number placements. Each step involves
selecting an empty cell and placing a valid number in it. As we make selections, the set of valid moves
for remaining cells may change.

# Action Prior System Instruction and User Request
## System Instruction
−−−−−−−−−−−−− Insert game rules here −−−−−−−−−−−−−
Important considerations when evaluating possible actions:
1. Howactions might create naked singles or hidden singles in other cells
2. Actions targeting cells with few remaining alternatives
3. How actions may constrain multiple other cells simultaneously
4. How actions contribute to a balanced distribution of numbers across the board
5. Whether actions might lead to contradictions or cells with no legal moves
Your task is to evaluate the possible actions in the current state, scoring them based on
how likely they are to help solve the Sudoku puzzle. The scores should form a probability
distribution over the actions (sum to 1.0) and be returned as a dictionary mapping action indices
to scores.
Example {grid_size} × {grid_size} Sudoku Board
{example_board}
Example Possible Actions
{example_prior_actions}
Example Final Answer
{"operation_scores" : {example_operation_scores}}

## User Request
Current {grid_size} × {grid_size} Sudoku Board
{current_board}
Possible Actions
{action_list}
Evaluate each action based on how it creates constraints, identifies singles, minimizes branch
ing, and maintains a balanced distribution of numbers as described in your instructions.
Assign a probability to each possible action based on how likely it is to lead to a solution of the
Sudoku puzzle. The scores should sum to 1.0, representing a probability distribution over the
actions.
Your response must include a valid JSON object, enclosed in a boxed, with an
operation_scores field containing a dictionary mapping action indices to scores, formatted
as follows:
{"operation_scores" :< dictionary_of_scores >}
Replace <dictionary_of_scores> with a dictionary mapping action indices to scores that
MUST sum to 1.0