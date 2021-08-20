from gym.envs.registration import register

register(
    id='ao-v0',
    entry_point='ao_env.envs:AdaptiveOptics',
)

register(
    id='ao-v0-bright',
    entry_point='ao_env.envs:AdaptiveOpticsBright',
)
