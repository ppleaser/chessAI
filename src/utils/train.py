from .buffer import ReplayBuffer

def train_neural_net(model, replay_buffer, batch_size):

    replay_buffer_for_training = ReplayBuffer(10000)
    replay_buffer_for_training.buffer = list(replay_buffer)

    states, actions, rewards, next_states, dones = replay_buffer_for_training.sample(batch_size)

    target_values = rewards + (1 - dones) * model.predict([next_states, actions])[:, 0]
    target = model.predict([states, actions])
    target[:, 0] = target_values

    model.train_on_batch([states, actions], target)
    print("Model was trained successfully")
    return model