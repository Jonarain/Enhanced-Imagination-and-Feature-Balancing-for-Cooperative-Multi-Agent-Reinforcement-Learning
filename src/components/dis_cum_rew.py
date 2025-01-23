import torch
def discounted_cumulative_rewards(rewards, gamma):
    # Get the shape of the rewards tensor
    batch_size, num_steps, _ = rewards.shape

    # Initialize the result tensor
    discounted_rewards = torch.zeros_like(rewards)

    # Iterate over steps in reverse order
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            discounted_rewards[:, t, 0] = rewards[:, t, 0]
        else:
            discounted_rewards[:, t, 0] = rewards[:, t, 0] + gamma * discounted_rewards[:, t + 1, 0]

    return discounted_rewards


def test_discounted_cumulative_rewards():
    # Set up a simple rewards tensor and known discount factor
    rewards = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]],
                            [[1.0], [2.0], [3.0], [4.0], [0.]]])
    gamma = 0.9

    # Manually calculate the expected discounted cumulative rewards
    expected_rewards = torch.tensor([[[11.4265], [11.585], [10.65], [8.5], [5.0]],
                                     [[8.146], [7.94], [6.6], [4.0], [0]]])

    # Calculate the discounted cumulative rewards using the function
    discounted_rewards = discounted_cumulative_rewards(rewards, gamma)

    # Check if the calculated rewards match the expected rewards
    assert torch.allclose(discounted_rewards, expected_rewards, atol=1e-3), \
        f"Expected {expected_rewards}, but got {discounted_rewards}"

    print("Test passed!")


# Run the test case
test_discounted_cumulative_rewards()