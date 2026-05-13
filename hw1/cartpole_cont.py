import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import importlib


class CartPoleContEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, initial_theta=0.0):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5
        self.tau = 0.01  # seconds between state updates
        self.initial_theta = initial_theta
        self.planning_steps = 600

        # Angle at which to fail the episode
        self.theta_threshold_radians = np.pi / 8.0
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.array([100.0])
        self.action_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        # pygame-based renderer (used when Gym's legacy rendering utilities are unavailable)
        self._pygame_screen = None
        self._pygame_clock = None
        self.state = None
        self.planning_steps_counter = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_accelerations(self, state, action):
        x, x_dot, theta, theta_dot = state
        force = action
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        temp1 = force - self.masspole * self.length * theta_dot * theta_dot * sin_theta
        temp2 = self.masscart + self.masspole * sin_theta * sin_theta

        theta_acc = (temp1 * cos_theta + self.gravity * sin_theta * (self.masspole + self.masscart)) / (self.length * temp2)
        x_acc = (temp1 + self.gravity * self.masspole * sin_theta * cos_theta) / temp2

        return theta_acc, x_acc

    def get_state_change(self, state, action):
        theta_acc, x_acc = self._compute_accelerations(state, action)

        state_change = np.array([state[1], x_acc, state[3], theta_acc])
        state_change *= self.tau

        return np.array(self.state) + state_change

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.state = tuple((self.get_state_change(self.state, action[0])).tolist())

        self.planning_steps_counter += 1
        done = self.planning_steps_counter >= self.planning_steps

        reward = 1.0 if np.abs(self.state[2]) < self.theta_threshold_radians else -1.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.planning_steps_counter = 0
        self.state = np.array([0.0, 0.0, self.initial_theta, 0.0])
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        # Try Gym's legacy viewer-based rendering if it exists; otherwise fall back to pygame.
        if self.viewer is None and self._pygame_screen is None:
            try:
                rendering = importlib.import_module('gym.envs.classic_control.rendering')
            except ModuleNotFoundError:
                rendering = None

            if rendering is not None:
                self.viewer = rendering.Viewer(screen_width, screen_height)
                l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
                axleoffset = cartheight / 4.0
                cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                self.carttrans = rendering.Transform()
                cart.add_attr(self.carttrans)
                self.viewer.add_geom(cart)
                l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
                pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                pole.set_color(.8, .6, .4)
                self.poletrans = rendering.Transform(translation=(0, axleoffset))
                pole.add_attr(self.poletrans)
                pole.add_attr(self.carttrans)
                self.viewer.add_geom(pole)
                self.axle = rendering.make_circle(polewidth / 2)
                self.axle.add_attr(self.poletrans)
                self.axle.add_attr(self.carttrans)
                self.axle.set_color(.5, .5, .8)
                self.viewer.add_geom(self.axle)
                self.track = rendering.Line((0, carty), (screen_width, carty))
                self.track.set_color(0, 0, 0)
                self.viewer.add_geom(self.track)
                self._pole_geom = pole
            else:
                # pygame fallback (works on modern Gym setups that removed the legacy rendering module)
                try:
                    import pygame
                except ImportError as e:
                    raise ImportError(
                        "Rendering requires either Gym's legacy classic_control.rendering module "
                        "or the 'pygame' package. Install pygame via: pip install pygame"
                    ) from e

                pygame.init()
                self._pygame_screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption('CartPoleContEnv')
                self._pygame_clock = pygame.time.Clock()

        if self.state is None: return None

        # --- Path 1: legacy Gym viewer ---
        if self.viewer is not None:
            # Edit the pole polygon vertex
            pole = self._pole_geom
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole.v = [(l, b), (l, t), (r, t), (r, b)]

            x = self.state
            cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
            cartx %= screen_width
            self.carttrans.set_translation(cartx, carty)
            self.poletrans.set_rotation(-x[2])
            return self.viewer.render(return_rgb_array=mode == 'rgb_array')

        # --- Path 2: pygame fallback ---
        import pygame

        # Handle window events so the OS doesn't think the app hung.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        # Background
        self._pygame_screen.fill((255, 255, 255))

        # Track
        pygame.draw.line(self._pygame_screen, (0, 0, 0), (0, carty), (screen_width, carty), 2)

        x, x_dot, theta, theta_dot = self.state
        cartx = x * scale + screen_width / 2.0
        cartx %= screen_width

        # Cart rectangle
        cart_left = cartx - cartwidth / 2.0
        cart_top = carty - cartheight / 2.0
        cart_rect = pygame.Rect(int(cart_left), int(cart_top), int(cartwidth), int(cartheight))
        pygame.draw.rect(self._pygame_screen, (0, 0, 0), cart_rect, 0)

        # Pole as a thick line (simple + robust)
        axle_x = cartx
        axle_y = carty - cartheight / 4.0
        tip_x = axle_x + polelen * math.sin(theta)
        tip_y = axle_y - polelen * math.cos(theta)
        pygame.draw.line(
            self._pygame_screen,
            (204, 153, 102),
            (int(axle_x), int(axle_y)),
            (int(tip_x), int(tip_y)),
            int(polewidth),
        )

        # Axle
        pygame.draw.circle(self._pygame_screen, (128, 128, 204), (int(axle_x), int(axle_y)), int(polewidth / 2))

        if mode == 'human':
            pygame.display.flip()
            self._pygame_clock.tick(self.metadata.get('video.frames_per_second', 50))
            return None
        elif mode == 'rgb_array':
            # (W, H, C) -> (H, W, C)
            arr = pygame.surfarray.array3d(self._pygame_screen)
            return np.transpose(arr, (1, 0, 2))
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self._pygame_screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._pygame_screen = None
            self._pygame_clock = None


if __name__ == '__main__':
    env = CartPoleContEnv()
    # run no force
    env.reset()
    env.render()
    is_done = False
    while not is_done:
        _, r, is_done, _ = env.step(np.array([0.0], dtype=np.float32))
        env.render()
        print(r)
    # run random forces
    env.reset()
    env.render()
    is_done = False
    while not is_done:
        _, r, is_done, _ = env.step(env.action_space.sample())  # take a random action
        env.render()
        print(r)
    env.close()
