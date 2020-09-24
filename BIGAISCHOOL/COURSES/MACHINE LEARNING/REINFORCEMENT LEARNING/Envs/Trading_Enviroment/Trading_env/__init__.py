from gym.envs.registration import register

register(
    id='Trading_env-v0',
    entry_point='Trading_env.envs:TradingEnv',
)