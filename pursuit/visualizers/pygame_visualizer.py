import collections
import sys, pygame
from threading import Thread

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class PygameVisualizer(object):
    def __init__(self, width=320, height=240, agent_colors=(255, 0, 0), prey_colors=(0, 255, 0)):
        self.size = self.width, self.height = width, height
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        assert((isinstance(agent_colors, tuple) or isinstance(agent_colors, list)) and len(agent_colors) > 0)
        assert((isinstance(prey_colors, tuple) or isinstance(prey_colors, list)) and len(prey_colors) > 0)
        if isinstance(agent_colors[0], int):
            agent_colors = (agent_colors, )
        if isinstance(prey_colors[0], int):
            prey_colors = (prey_colors,)

        self.agent_color = agent_colors
        self.prey_color = prey_colors
        self.state = None
        self.thread = None

    def start(self, state):
        self.state = state
        self.thread = Thread(target=self.draw, args=())
        self.thread.start()

    def update(self, state, actions, next_state, reward):
        self.state = next_state

    def end(self):
        pass

    def draw(self):
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.screen.fill(WHITE)
            wx, wy = self.state.world_size
            xstep, ystep = self.width / wx, self.height / wy
            padding = 0.5 # value between 0 and 1

            for x in range(wx - 1):
                pygame.draw.line(self.screen, BLACK, ((x+1)*xstep, 0), ((x+1)*xstep, self.height))

            for y in range(wy - 1):
                pygame.draw.line(self.screen, BLACK, (0, (y+1)*ystep), (self.width, (y+1)*ystep))

            agents = self.state.agent_positions
            preys = self.state.prey_positions

            for i, (x, y) in enumerate(agents):
                color = self.agent_color if len(self.agent_color) == 1 else self.agent_color[i]

                pygame.draw.rect(self.screen, color,
                                 pygame.Rect((x+padding/2)*xstep, (y+padding/2)*ystep,
                                             xstep-padding*xstep, ystep-padding*ystep))

            for i, (x, y) in enumerate(preys):
                color = self.prey_color if len(self.prey_color) == 1 else self.prey_color[i]

                pygame.draw.rect(self.screen, color,
                                 pygame.Rect((x+padding/2)*xstep, (y+padding/2)*ystep,
                                             xstep-padding*xstep, ystep-padding*ystep))


            pygame.display.flip()

