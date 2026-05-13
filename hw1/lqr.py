import numpy as np
from cartpole_cont import CartPoleContEnv


class _Float32Action:
    def __init__(self, value):
        # Store as numpy scalar so downstream np.array([...]) keeps float32 dtype.
        self._value = np.float32(value)

    def item(self, *args, **kwargs):
        # Match numpy's `.item(...)` API used in the starter code.
        return self._value


class _GainForGymBox:
    """Wraps a (1x4) gain so `K * x` yields a float32, in-range scalar via `.item(0)`.

    This lets the provided main code do:
        actual_action = (K * state).item(0)
        actual_action = np.array([actual_action])
    and still satisfy `action_space.contains()` (dtype float32).
    """

    def __init__(self, K, low, high):
        self._K = K
        self._low = float(low)
        self._high = float(high)

    @property
    def shape(self):
        return self._K.shape

    def __mul__(self, other):
        res = self._K * other
        # `res` is expected to be shape (1,1)
        value = float(res.item(0))
        value = max(self._low, min(self._high, value))
        return _Float32Action(value)


def get_A(cart_pole_env):
    '''
    create and returns the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau

    A_c = np.matrix([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, (pole_mass / cart_mass) * g, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, (g / pole_length) * (1.0 + pole_mass / cart_mass), 0.0],
    ])
    return np.matrix(np.eye(4)) + A_c * dt


def get_B(cart_pole_env):
    '''
    create and returns the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau

    B_c = np.matrix([
        [0.0],
        [1.0 / cart_mass],
        [0.0],
        [1.0 / (pole_length * cart_mass)],
    ])
    return B_c * dt


def find_lqr_control_input(cart_pole_env):
    '''
    implements the LQR algorithm
    :param cart_pole_env: to extract all the relevant constants
    :return: a tuple (xs, us, Ks). xs - a list of (predicted) states, each element is a numpy array of shape (4,1).
    us - a list of (predicted) controls, each element is a numpy array of shape (1,1). Ks - a list of control transforms
    to map from state to action, np.matrix of shape (1,4).
    '''
    assert isinstance(cart_pole_env, CartPoleContEnv)

    # TODO - you first need to compute A and B for LQR
    A = get_A(cart_pole_env)
    B = get_B(cart_pole_env)

    # TODO - Q and R should not be zero, find values that work, hint: all the values can be <= 1.0
    Q = np.matrix([
        [1, 0, 0, 0],
        [0, 0.01, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0.01]
    ])

    R = np.matrix([3])

    # TODO - you need to compute these matrices in your solution, but these are not returned.
    Ps = [None] * (cart_pole_env.planning_steps+1)
    Ps[-1] = Q
    for i in range(cart_pole_env.planning_steps - 1, -1, -1):
        Ps[i] = Q + A.T * Ps[i+1] * A - A.T * Ps[i+1] * B * np.linalg.inv(R + B.T * Ps[i+1] * B) * B.T * Ps[i+1] * A

    # TODO - these should be returned see documentation above
    # Internally we keep numeric gain matrices for planning, but we return wrapped gains
    # that produce float32 clipped actions when used in the provided main loop.
    K_mats = []
    Ks = []
    low = cart_pole_env.action_space.low.item(0)
    high = cart_pole_env.action_space.high.item(0)
    action_dtype = cart_pole_env.action_space.dtype
    for i in range(cart_pole_env.planning_steps):
        K = -np.linalg.inv(R + B.T * Ps[i+1] * B) * B.T * Ps[i+1] * A
        K_mats.append(K)
        Ks.append(_GainForGymBox(K, low, high))
    us = []
    xs = [np.expand_dims(cart_pole_env.state, 1)]
    for i in range(cart_pole_env.planning_steps):
        u = K_mats[i] * xs[i]
        u_scalar = float(u.item(0))
        u_scalar = max(low, min(high, u_scalar))
        # Store as float32 so that `us[i].item(0)` is a float32 scalar.
        u_clipped = np.array([[u_scalar]], dtype=action_dtype)
        us.append(u_clipped)
        xs.append(A * xs[i] + B * u_clipped)

    print("length of xs: {}, length of us: {}, length of Ks: {}, length of Ps: {}, time horizon: {}".format(len(xs), len(us), len(Ks), len(Ps), cart_pole_env.planning_steps))
    assert len(xs) == cart_pole_env.planning_steps + 1, "if you plan for x states there should be X+1 states here"
    assert len(us) == cart_pole_env.planning_steps, "if you plan for x states there should be X actions here"
    for x in xs:
        assert x.shape == (4, 1), "make sure the state dimension is correct: should be (4,1)"
    for u in us:
        assert u.shape == (1, 1), "make sure the action dimension is correct: should be (1,1)"
    return xs, us, Ks


def print_diff(iteration, planned_theta, actual_theta, planned_action, actual_action):
    print('iteration {}'.format(iteration))
    print('planned theta: {}, actual theta: {}, difference: {}'.format(
        planned_theta, actual_theta, np.abs(planned_theta - actual_theta)
    ))
    print('planned action: {}, actual action: {}, difference: {}'.format(
        planned_action, actual_action, np.abs(planned_action - actual_action)
    ))


def run_lqr_episode_and_log_theta(initial_theta, render=False):
    """Run one episode with LQR control and return theta values over time."""
    env = CartPoleContEnv(initial_theta=initial_theta)
    actual_state = env.reset()
    if render:
        env.render()

    xs, us, Ks = find_lqr_control_input(env)

    thetas = [float(actual_state[2])]
    is_done = False
    iteration = 0
    while not is_done:
        # Use the planned open-loop control sequence `us`.
        actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
        # Make sure the action matches the env's Box requirements.
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_state, reward, is_done, _ = env.step(np.array([actual_action], dtype=env.action_space.dtype))
        thetas.append(float(actual_state[2]))
        if render:
            env.render()
        iteration += 1

    env.close()
    return thetas


def plot_theta_over_time():
    """Plot theta(t) for initial theta = 0.1π, 0.16π, 0.32π."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    thetas_01pi = run_lqr_episode_and_log_theta(initial_theta=0.1 * np.pi, render=False)
    thetas_016pi = run_lqr_episode_and_log_theta(initial_theta=0.16 * np.pi, render=False)
    thetas_032pi = run_lqr_episode_and_log_theta(initial_theta=0.32 * np.pi, render=False)

    plt.figure()
    plt.plot(thetas_01pi, label='initial theta = 0.1π')
    plt.plot(thetas_016pi, label='initial theta = 0.16π')
    plt.plot(thetas_032pi, label='initial theta = 0.32π')
    plt.xlabel('time step')
    plt.ylabel('theta (rad)')
    plt.title('CartPole LQR: theta over time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import sys

    # Default: plot theta over time (no rendering window).
    # Use `--render` to watch the cart move.
    if '--render' not in sys.argv:
        plot_theta_over_time()
        raise SystemExit(0)

    env = CartPoleContEnv(initial_theta=np.pi * 0.15)
    # the following is an example to start at a different theta
    # env = CartPoleContEnv(initial_theta=np.pi * 0.25)

    # print the matrices used in LQR
    print('A: {}'.format(get_A(env)))
    print('B: {}'.format(get_B(env)))

    # start a new episode
    actual_state = env.reset()
    env.render()
    # use LQR to plan controls
    xs, us, Ks = find_lqr_control_input(env)
    # run the episode until termination, and print the difference between planned and actual
    is_done = False
    iteration = 0
    is_stable_all = []
    while not is_done:
        # print the differences between planning and execution time
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
        print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
        # apply action according to actual state visited
        # make action in range
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_action = np.array([actual_action], dtype=env.action_space.dtype)
        actual_state, reward, is_done, _ = env.step(actual_action)
        is_stable = reward == 1.0
        is_stable_all.append(is_stable)
        env.render()
        iteration += 1
    env.close()
    # we assume a valid episode is an episode where the agent managed to stabilize the pole for the last 100 time-steps
    valid_episode = np.all(is_stable_all[-100:])
    # print if LQR succeeded
    print('valid episode: {}'.format(valid_episode))

