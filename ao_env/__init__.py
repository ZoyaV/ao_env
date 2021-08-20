from gym.envs.registration import register

register(
    id='ao-v0',
    entry_point='ao_env.envs:AdaptiveOptics',
)

register(
    id='aoebright-v0',
    entry_point='ao_env.envs:AdaptiveOpticsBright',
)

