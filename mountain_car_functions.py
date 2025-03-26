import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import random
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.colors as mcolors

def optimal_transition_function(position,velocity,action_input = None, N=1):
    ''' 
    Input: Start state (position, velocity) 
    Output: Optimal N step trajectory given the start state 
    Trajectory = [position, velocity, next_position, next_velocity, reward, action] (S,A,S',R) * N
    '''
    trajectory = [] 
    for i in range(N):
        action = -1 if velocity < 0 else 1 #Optimal Policy
        if action_input is not None:
            action = action_input
        next_velocity = velocity + (0.001 * action - 0.0025 * np.cos(3 * position))
        next_velocity = np.clip(next_velocity, -0.07, 0.07)
        next_position = position + next_velocity
        next_position = np.clip(next_position, -1.2, 0.6)
        if (next_position == -1.2 and next_velocity < 0): next_velocity = 0
        reward = -(next_position - 0.5) ** 2
        trajectory.append(torch.tensor([position, velocity,next_position, next_velocity, reward, action], dtype=torch.float32))
        position = next_position
        velocity = next_velocity
    return trajectory

def get_data(n):
    positions = np.linspace(-1.2,0.4,100) 
    velocities = np.linspace(-0.07,0.07,100)
    all_data = []
    for position in positions:
        for velocity in velocities:
            trajectory = optimal_transition_function(position,velocity,N=n)
            all_data.append(trajectory)

    #all_data = np.array(all_data)
    np.random.shuffle(all_data)
    train_data = all_data[:8000]
    test_data = all_data[8000:]
    return train_data, test_data

class StateEncoder(nn.Module): 
    # 2 -> discrete_states  
    def __init__(self, discrete_states):
        super(StateEncoder, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, discrete_states)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Softmax final layer 
        x = F.softmax(self.fc2(x), dim=0)
        return x
    
class StateDecoder(nn.Module):
    # discrete_states -> 2
    def __init__(self, discrete_states):
        super(StateDecoder, self).__init__()
        self.fc1 = nn.Linear(discrete_states, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Normalize transition matrix before calculation
class TransitionModel(nn.Module):
    def __init__(self, discrete_states):
        super(TransitionModel, self).__init__()
        self.state_encoder = StateEncoder(discrete_states) 
        self.transition_matrix = nn.Parameter(torch.rand(2, discrete_states, discrete_states))
        self.state_decoder = StateDecoder(discrete_states)

    # trajectory is curr state[:2], next state[2:4], reward, action
    def forward(self, trajectory):
        prediction = []
        decoded_states = []
        curr_state = trajectory[0][:2]
        curr_state = self.state_encoder(curr_state)

        for timestep in trajectory:  # Really all we need here are the actions
            action = int(timestep[5])
            action = 0 if action == -1 else 1
            transition_matrix = self.transition_matrix[action]
            transition_matrix = F.softmax(transition_matrix, dim=0) ####Seems to make things worse? Commented out for report.ipynb
            # Transition step
            predicted_next_state = torch.matmul(transition_matrix, curr_state)
            prediction.append(predicted_next_state)

            # Decode state
            decoded_state = self.state_decoder(curr_state)
            decoded_states.append(decoded_state)

            # Update current state
            curr_state = predicted_next_state

        return prediction, decoded_states
    
def train(model, optimizer, train_data, test_data, epochs):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Track cumulative losses
        total_train_loss = 0.0
        total_test_loss = 0.0

        # Training loop
        for i in random.sample(range(len(train_data)), len(train_data)):
            trajectory = train_data[i]
            optimizer.zero_grad()
            prediction, decode = model(trajectory) 
            true_initial_states = torch.stack([trajectory[0][:2] for _ in range(len(prediction))])
            true_next_states = torch.stack([trajectory[i][2:4] for i in range(len(prediction))])
            encoded_next_states = torch.stack([model.state_encoder(trajectory[i][2:4]) for i in range(len(prediction))])
            loss = F.mse_loss(torch.stack(prediction), encoded_next_states) + F.mse_loss(torch.stack(decode), true_initial_states) + F.mse_loss(model.state_decoder(torch.stack(prediction)), true_next_states)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        #print("Epoch:", epoch, "Training Loss:", total_train_loss/len(train_data))
        train_losses.append(total_train_loss/len(train_data))

        # Testing loop
        with torch.no_grad():
            for i in range(len(test_data)):
                trajectory = test_data[i]
                prediction, decode = model(trajectory)
                true_initial_states = torch.stack([trajectory[0][:2] for _ in range(len(prediction))])
                true_next_states = torch.stack([trajectory[i][2:4] for i in range(len(prediction))])
                encoded_next_states = torch.stack([model.state_encoder(trajectory[i][2:4]) for i in range(len(prediction))])
                loss = F.mse_loss(torch.stack(prediction), encoded_next_states) + F.mse_loss(torch.stack(decode), true_initial_states) + F.mse_loss(model.state_decoder(torch.stack(prediction)), true_next_states)
                total_test_loss += loss.item()
           # print("Epoch:", epoch, "Test Loss:", total_test_loss/len(test_data))
            test_losses.append(total_test_loss/len(test_data))
    plt.plot(train_losses[2:])
    plt.plot(test_losses[2:])
    plt.legend(["Train Loss", "Test Loss"])
    plt.title("Results epoch 2+")
    plt.show()
    
def test_trajectory(model, initial_state, num_steps):
    state = initial_state
    #print(state)
    true_state = state
    model_trajectory = [state]
    true_trajectory = [state]
    for i in range(num_steps):
        action = 0 if state[1].item() < 0 else 1
        enc_state = model.state_encoder(torch.tensor(state, dtype=torch.float32))
        model_next_state = torch.matmul(model.transition_matrix[action], enc_state)
        next_state = model.state_decoder(model_next_state)
        #next_state = model.state_decoder(next_state)
        state = next_state
        model_trajectory.append(state.detach().numpy())
        true_next_state = optimal_transition_function(true_state[0], true_state[1], action_input=action, N=1)[0][2:4]
        true_state = true_next_state
        true_trajectory.append(true_state)
    #graph real vs true
    model_trajectory = np.array(model_trajectory)
    true_trajectory = np.array(true_trajectory)
    plt.figure(figsize=(8, 6))
    plt.plot(model_trajectory[:, 0], model_trajectory[:, 1], label="Model Trajectory", marker="o")
    plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], label="True Trajectory", marker="s", linestyle="dashed")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title("Model vs. True Trajectory")
    plt.legend()
    plt.grid()
    plt.show()

def plot_categorical_states(train_data, tmodel):
    """
    Plots categorical state predictions with dynamic and constant alpha values.

    Parameters:
    - train_data: List of state-action pairs.
    - tmodel: Model with a state encoder that outputs category probabilities.
    """
    argmax_states = []
    alphas = []  # List to store dynamic alpha values

    # Process state positions, argmax values, and max probabilities
    for i in train_data: 
        state = torch.tensor(i[0][:2]).clone().detach().float()
        probabilities = tmodel.state_encoder(state)
        argmax_value = probabilities.argmax().item()
        max_prob = probabilities.max().item()  # Get max probability for dynamic alpha

        argmax_states.append([np.array(state), argmax_value])
        alphas.append(max_prob)  # Use this for alpha scaling

    # Convert to numpy arrays for plotting
    x = np.array([s[0][0] for s in argmax_states])
    y = np.array([s[0][1] for s in argmax_states])
    colors = np.array([s[1] for s in argmax_states])
    alphas = np.array(alphas)  # Convert alpha list to numpy array

    # Ensure color categories are mapped from 1-9 regardless of missing categories
    num_categories = 9
    cmap = plt.get_cmap('tab10', num_categories)  # Use tab10 for discrete colors
    norm = mcolors.Normalize(vmin=1, vmax=num_categories)  # Fix range from 1-9

    # Create side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    scatter1 = axs[0].scatter(x, y, c=colors, cmap=cmap, norm=norm, edgecolors='k', alpha=alphas)
    axs[0].set_title("Alpha = Max Probability")
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Velocity")
    fig.colorbar(scatter1, ax=axs[0], label="Argmax Category", ticks=range(1, num_categories+1))

    scatter2 = axs[1].scatter(x, y, c=colors, cmap=cmap, norm=norm, edgecolors='k', alpha=0.75)
    axs[1].set_title("Alpha = 0.75 (Constant)")
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Velocity")
    fig.colorbar(scatter2, ax=axs[1], label="Argmax Category", ticks=range(1, num_categories+1))

    plt.tight_layout()
    plt.show()

def plot_categorical_states_plotly(train_data, tmodel):
    """
    Creates an interactive Plotly scatter plot for categorical state predictions.

    Parameters:
    - train_data: List of state-action pairs.
    - tmodel: Model with a state encoder that outputs category probabilities.
    """
    argmax_states = []
    alphas = []  # List to store dynamic alpha values
    max_probs = []  # Store max probability for hover info

    # Process state positions, argmax values, and max probabilities
    for i in train_data: 
        state = torch.tensor(i[0][:2]).clone().detach().float()
        probabilities = tmodel.state_encoder(state)
        argmax_value = probabilities.argmax().item()
        max_prob = probabilities.max().item()  # Get max probability for dynamic alpha

        argmax_states.append([np.array(state), argmax_value])
        alphas.append(max_prob)  # Use max probability for alpha scaling
        max_probs.append(max_prob)  # Store for hover display

    # Convert to Pandas DataFrame for Plotly
    df = pd.DataFrame({
        "x": [s[0][0] for s in argmax_states],
        "y": [s[0][1] for s in argmax_states],
        "argmax": [s[1] for s in argmax_states],
        "max_prob": max_probs,
        "alpha": alphas  # Opacity tied to probability
    })

    # Ensure all categories 1-9 are represented even if missing
    num_categories = 9
    all_categories = list(range(1, num_categories + 1))  # [1,2,3,...,9]

    # Use the same color scheme as Matplotlib
    tab10_colors = plt.get_cmap('tab10', num_categories).colors  
    color_map = {str(i): f"rgb{tuple(int(255 * c) for c in tab10_colors[i-1])}" for i in all_categories}  

    # Map missing categories into dataset (if any category is missing, it still appears in legend)
    df["argmax"] = df["argmax"].astype(str)  # Ensure categorical labels for proper ordering

    # Create scatter plot with Plotly
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="argmax",
        color_discrete_map=color_map,  # Use the fixed color mapping
        opacity=df["alpha"],  # Dynamic opacity
        hover_data={"x": True, "y": True, "argmax": True, "max_prob": True},  # Show on hover
        category_orders={"argmax": [str(i) for i in all_categories]},  # Force color scale to show all categories
    )

    # Update layout for better readability
    fig.update_layout(
        title="2D State Plot with Argmax as Color (Alpha = Max Probability)",
        xaxis_title="Position",
        yaxis_title="Velocity",
        legend_title="Argmax Category"
    )

    # Show the interactive plot
    fig.show()

def do_all(train_data,test_data,discrete_states, learning_rate, epochs):
    tmodel = TransitionModel(discrete_states)
    optimizer = optim.Adam(tmodel.parameters(), lr=learning_rate)
    train(tmodel, optimizer, train_data, test_data, epochs)
    test_state = test_data[0][0][:2]
    test_trajectory(tmodel, test_state, 50)
    plot_categorical_states(train_data, tmodel)
    plot_categorical_states_plotly(train_data, tmodel)
    # print("Test Data Categorical States:")
    # plot_categorical_states(test_data, tmodel)
    # plot_categorical_states_plotly(test_data, tmodel)


