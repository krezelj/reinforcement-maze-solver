import numpy as np
import cv2 as cv

# 3 - direction = opposite direction
UP = 0
RIGHT = 1
LEFT = 2
DOWN = 3

relative_direction = {
    UP: (-1, 0), 
    RIGHT: (0, 1),
    LEFT: (0, -1),
    DOWN: (1, 0),
}

class Maze():

    CELL_SIZE = 30
    CELL_MARGIN = 2

    __slots__ = ['cells', 'shape']

    def __init__(self, shape) -> None:
        self.shape = shape
        self.__generate()

    @classmethod
    def rel2abs(cls, row, col, direction):
        direction_tuple = relative_direction[direction]
        return row + direction_tuple[0], col + direction_tuple[1]

    def get_neighbours(self, row, col):
        connections = self.cells[row, col]
        neighbours = []
        for direction in [UP, RIGHT, DOWN, LEFT]:
            if connections[direction]:
                neighbours.append(Maze.rel2abs(row, col, direction))
        return neighbours


    def __generate(self):
        
        self.cells = np.zeros(shape=self.shape + (4,)) # 4 directions, up right down left
        visited = np.zeros(shape=self.shape)

        def get_available_directions(row, col):
            directions = []
            if row > 0 and not visited[row - 1, col]:
                directions.append(UP)
            if row < self.shape[0] - 1 and not visited[row + 1, col]:
                directions.append(DOWN)
            if col > 0 and not visited[row, col - 1]:
                directions.append(LEFT)
            if col < self.shape[1] - 1 and not visited[row, col + 1]:
                directions.append(RIGHT)
            return directions

        stack = [(0, 0)]
        visited[(0, 0)] = 1

        while len(stack) > 0:
            current_cell = stack[-1]
            visited[current_cell] = 1

            available_directions = get_available_directions(*current_cell)

            if len(available_directions) > 0:
                direction = np.random.choice(available_directions, size=1)[0]
                next_cell = Maze.rel2abs(*current_cell, direction)

                self.cells[current_cell + (direction,)] = 1
                self.cells[next_cell + (3 - direction,)] = 1

                stack.append(next_cell)
            else:
                stack.pop()
            

    def render(self, path=None, goal=None, animate=False):
        def top_left(row, col):
            # reversed because opencv uses x, y order and not row, col
            return (col * self.CELL_SIZE + self.CELL_MARGIN, row * self.CELL_SIZE + self.CELL_MARGIN)

        def bottom_right(row, col):
            # reversed because opencv uses x, y order and not row, col
            return ((col + 1) * self.CELL_SIZE - self.CELL_MARGIN, (row + 1) * self.CELL_SIZE - self.CELL_MARGIN)

        def get_cell_colour(i):
            alpha = i / len(path)
            return (int(255 * alpha), 0, int(255 * (1-alpha)))

        img = np.zeros((self.shape[0] * self.CELL_SIZE, self.shape[1] * self.CELL_SIZE) + (3,), dtype=np.uint8)

        # draw cells
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                p1 = top_left(row, col)
                p2 = bottom_right(row, col)
                cv.rectangle(img, p1, p2, color=(255, 255, 255), thickness=-1)

        # draw goal
        if goal is not None:
            p1 = top_left(*goal)
            p2 = bottom_right(*goal)
            cv.rectangle(img, p1, p2, color=(0, 255, 0), thickness=-1)

        # draw connections
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                c = (254, 231, 222)
                if col < self.shape[1] - 1 and self.cells[row, col, RIGHT]:
                    p1 = bottom_right(row, col)
                    p2 = top_left(*Maze.rel2abs(row, col, RIGHT))
                    cv.rectangle(img, p1, p2, color=c, thickness=-1)
                if row < self.shape[0] - 1 and self.cells[row, col, DOWN]:
                    p1 = bottom_right(row, col)
                    p2 = top_left(*Maze.rel2abs(row, col, DOWN))
                    cv.rectangle(img, p1, p2, color=c, thickness=-1)

        # draw path
        if path is None:
            cv.imshow('maze', img)
            cv.waitKey(0)
            return
        
        # if path is not None
        for i, cell in enumerate(path):
            colour = get_cell_colour(i)
            p1 = top_left(*cell)
            p2 = bottom_right(*cell)
            cv.rectangle(img, p1, p2, color=colour, thickness=-1)
            if animate:
                cv.imshow('maze', img)
                if cv.waitKey(20) == ord('q'):
                    print("animation stopped")
                    animate = False
            
        cv.imshow('maze', img)
        cv.waitKey(2000)

            


def main():
    maze = Maze(shape=(20, 10))
    maze.render()


if __name__ == '__main__':
    main()