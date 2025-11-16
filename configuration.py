class Configuration:
    gamma=0.99
    lr=5e-5
    replay_capacity=10000
    replay_init_ratio=0.3
    train_env_steps=200000
    target_update_period=100
    eps_init=1.0
    eps_final=0.05
    eps_decrease_step=100000
    num_eval_episodes=5
    eval_period=500