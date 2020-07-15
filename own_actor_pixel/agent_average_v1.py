import torch
import torchx as tx

from models import Actor, Critic, CNNNetwork
from cnn_models import CNNActor, CNNCritic

import torch.nn as nn
import torch.nn.functional as F




# Building the whole Training Process into a class

class TD31v1(object):
    def __init__(self, state_dim, action_dim, actor_input_dim, args):
        input_dim = [3, 84, 84]
        self.actor = CNNActor(input_dim, state_dim, action_dim).to(args.device)
        self.actor_target = CNNActor(input_dim, state_dim, action_dim).to(args.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), args.lr_actor)
        self.critic = CNNCritic(input_dim, state_dim, action_dim).to(args.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), args.lr_critic)
        self.list_target_critic = []
        for c in range(args.num_q_target):
            critic_target = CNNCritic(input_dim, state_dim, action_dim).to(args.device)
            critic_target.load_state_dict(critic_target.state_dict())
            self.list_target_critic.append(critic_target)
        self.target_critic = CNNCritic(input_dim, state_dim, action_dim).to(args.device)
        self.target_critic.load_state_dict(self.target_critic.state_dict())
        conv_hidden_dim = 200
        conv_out_channels = [16, 32]
        conv_kernel_sizes = [8, 4]
        conv_strides = [4, 2]
        self.max_action = 1
        self.num_q_target = args.num_q_target
        self.update_counter = 0
        self.step = 0 
        self.currentQNet = 0
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.tau = args.tau 
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.device = args.device
        self.actor_clip_gradient = args.actor_clip_gradient
        print("Use ", self.device)
        if self.device == "cpu":
            print("No GPU found need to have gpu")
        #self.perception = CNNNetwork(input_dim, conv_hidden_dim, conv_channels=conv_out_channels, kernel_sizes=conv_kernel_sizes, strides=conv_strides).to(self.device)
        #self.perception_optimizer = torch.optim.Adam(self.perception.parameters(), args.lr_actor)

    def select_action(self, state):
        #state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        state = torch.Tensor(state).to(self.device)
        state = state.unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()
    
    
    def train(self, replay_buffer, writer, iterations):
        self.step += 1
        for it in range(iterations):
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(self.batch_size)
            state = torch.Tensor(batch_states).to(self.device).div_(255)
            next_state = torch.Tensor(batch_next_states).to(self.device).div_(255)
            # create vector 
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device)
            done = torch.Tensor(batch_dones).to(self.device)
            with torch.no_grad(): 
                #state = self.perception(state)
                #next_state = self.perception(next_state)
                # Step 5: From the next state s’, the Actor target plays the next action a’
                next_action = self.actor_target(next_state)
                # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
                noise = torch.Tensor(batch_actions).data.normal_(0, self.policy_noise).to(self.device)
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
                target_Q = 0
                for critic in self.list_target_critic:
                    target_Q1, target_Q2 = critic(next_state, next_action) 
                    target_Q += torch.min(target_Q1, target_Q2)
                target_Q *= 1./ self.num_q_target  
                # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
                target_Q = reward + ((1 - done) * self.discount * target_Q).detach()
            
            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(state, action) 
            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
        
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            #self.perception_optimizer.step()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            #self.perception_optimizer.zero_grad()
            #perceptrion_loss.backward(retain_graph=True)
            #self.perception_optimizer.step() 
            #for param in self.critic.parameters():
                #print(param.grad.data.sum())
            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % self.policy_freq == 0:
                # print("cuurent", self.currentQNet)
                actor_loss = -self.critic.Q1(state, self.actor(state.detach())).mean()
                self.actor_optimizer.zero_grad()
                #actor_loss.backward(retain_graph=True)
                actor_loss.backward()
                # clip gradient
                #self.actor.clip_grad_value(self.actor_clip_gradient)
                #torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.actor_clip_gradient)
                self.actor_optimizer.step()
                
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
     
                
    def hardupdate(self):
        self.update_counter +=1
        self.currentQNet = self.update_counter % self.num_q_target
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.target_critic.parameters(), self.list_target_critic[self.currentQNet].parameters()):
            target_param.data.copy_(param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
                
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
                
