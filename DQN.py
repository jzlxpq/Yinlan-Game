import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import clip
import random
from collections import deque
import numpy as np
import os
import torch.nn.functional as F
from PIL import Image
import math
from dataclasses import dataclass
from repeat_attention import SelfAttentionWithClustering

REPLAY_SIZE = 2000
tb_writer = SummaryWriter(log_dir="DQN_RUN/experiment_1")

# class CrossAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(CrossAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#
#         assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
#
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.out = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, sequence_a, sequence_b):
#         batch_size, seq_len_a, _ = sequence_a.size()
#         _, seq_len_b, _ = sequence_b.size()
#
#         # Linear transformations for query, key, and value
#         Q = self.query(sequence_a)  # (batch_size, seq_len_a, embed_dim)
#         K = self.key(sequence_b)  # (batch_size, seq_len_b, embed_dim)
#         V = self.value(sequence_b)  # (batch_size, seq_len_b, embed_dim)
#
#         # Split into multiple heads
#         Q = Q.view(batch_size, seq_len_a, self.num_heads, self.head_dim).transpose(1, 2)
#         K = K.view(batch_size, seq_len_b, self.num_heads, self.head_dim).transpose(1, 2)
#         V = V.view(batch_size, seq_len_b, self.num_heads, self.head_dim).transpose(1, 2)
#
#         # Scaled Dot-Product Attention
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         attention_weights = F.softmax(scores, dim=-1)
#         output = torch.matmul(attention_weights, V)
#
#         # Concatenate heads and apply final linear layer
#         output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_a, self.embed_dim)
#         output = self.out(output)
#
#         return output, attention_weights


@dataclass
class Config:
    # General parameters
    n_embd: int = 512  # 嵌入维度
    n_head: int = 8  # 注意力头数
    dropout: float = 0.0  # Dropout 概率
    bias: bool = True  # 是否使用 bias
    block_size: int = 512  # 序列长度

    # For input features
    image_dim: int = 512  # 图像嵌入维度
    text_dim: int = 512  # 文本嵌入维度

    # Projection dimensions
    n_embd_proj: int = 512  # 嵌入投影维度
    n_head_proj: int = 8  # 头数投影维度

    # Flash Attention
    use_flash_attention: bool = True  # 是否使用 Flash Attention


config = Config()


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Key, Query, Value projections for cross-attention
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, context):
        """
        x: (B, Tx, C) - Query sequence
        context: (B, Tc, C) - Key/Value sequence (e.g., encoder output)
        """
        B, Tx, C = x.size()
        Tc = context.size(1)

        # Compute Q, K, V
        q = self.q_proj(x).view(B, Tx, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k_proj(context).view(B, Tc, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.v_proj(context).view(B, Tc, self.n_head, C // self.n_head).transpose(1, 2)

        # Cross-attention computation

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        print(f"X:{x.shape}",f"Tc:{Tc}", f"att维度是{att.shape}")
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, Tx, C)
        att_mean = att.mean(dim=1)
        # Output projection
        y = self.resid_dropout(self.out_proj(y))
        return y, att_mean


output_data = {}


def get_activation(name):
    def hook(model, input, output):

        output_data[name] = output.detach()  # output type is tensor

    return hook


class CLIP_GAME(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = clip.load("ViT-B/32", device=self.device)
        self.tokenizer = clip.tokenize
        self.reward = RewardModel()

    def encode_image(self, image):
        """使用 CLIP 的 image encoder 编码图像"""
        image = self.processor(image).unsqueeze(0).to(self.device)  # 预处理并添加 batch 维度
        with torch.no_grad():
            
            image_features = self.model.encode_image(image)

        return image_features  # 输出为 (1, 512)

    def encode_text(self, action_list):
        """使用 CLIP 的 text encoder 编码动作文本"""
        text_inputs = clip.tokenize(action_list).to(self.device)  # Tokenize 文本
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
        return text_features  # 输出为 (N, 512)，N 为动作数量

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        return image_features, text_features


class RewardModel(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(RewardModel, self).__init__()

        # Transformer 多头注意力
        self.attention = CrossAttention(config)

        # MLP 计算奖励
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, image_features, text_features):
        """
        使用 Transformer Attention 计算相似性并生成奖励
        :param image_features: (1, 512) -> (1, 1, 512)
        :param text_features: (N, 512) -> (N, 1, 512)
        :return: 归一化奖励 (N,)
        """
        B, _ = image_features.size()
        N, _ = text_features.size()
        # print("text_features is", text_features.shape, "image_features is ", image_features.shape)
        text_features = text_features.unsqueeze(0)  # (N, 1, 512) 扩展序列长度维度
        image_features = image_features.unsqueeze(1)  # (B, 1, 512) 扩展序列长度维度
        # print("text_features is", text_features.shape, "image_features is ", image_features.shape)
        # Transformer Attention
        attn_output, att = self.attention(image_features, text_features)  # (N, 1, 512)
        print("att is ", att.shape)
        # MLP 计算奖励
        rewards = att.squeeze(0).squeeze(0)

        # 归一化奖励 (Softmax)
        rewards = (rewards - rewards.mean()) / rewards.std()

        return rewards


class ImageReconstructionMLP(nn.Module):
    def __init__(self, input_dim=512, output_dim=3*224*224, hidden_dims=None):
        """
        从 image_features 重构成图像的 MLP
        :param input_dim: 输入特征维度（例如 CLIP 的 512 维）+
        :param output_dim: 输出图像维度（例如 3x224x224=150528）
        :param hidden_dims: 隐藏层维度列表
        """
        super(ImageReconstructionMLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 2048, 4096]
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 构建 MLP 层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # 使用 ReLU 激活函数
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))  # 最后一层映射到图像像素空间
        layers.append(nn.Tanh())  # 使用 Tanh 将输出限制在 [-1, 1] 范围内

        self.mlp = nn.Sequential(*layers)

    def forward(self, image_features):
        """
        前向传播
        :param image_features: 输入特征 (batch_size, input_dim)
        :return: 重构的图像 (batch_size, 3, 224, 224)
        """
        batch_size = image_features.size(0)
        # 通过 MLP 生成图像像素
        reconstructed_pixels = self.mlp(image_features)  # (batch_size, output_dim)
        # 将像素 reshape 为图像形状
        reconstructed_image = reconstructed_pixels.view(batch_size, 3, 224, 224)  # (batch_size, 3, 224, 224)
        return reconstructed_image


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def popleft(self):
        self.buffer.popleft()


class DQNAgent:
    def __init__(self, num_actions, action_texts, gamma=0.99, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.995,
                 model_dir="models"):

        self.action_texts = action_texts
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = float(epsilon)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_dir = model_dir
        # 创建模型目录
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.encoder_net = CLIP_GAME().to(self.device)
        self.policy_net = RewardModel().to(self.device)
        self.target_net = RewardModel().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(10000)
        self.self_attention = SelfAttentionWithClustering(512,1).to(self.device)

    def save_model(self, episode):
        """保存模型"""
        model_path = os.path.join(self.model_dir, f"dqn_model_{episode}.pth")
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self):
        """加载模型"""
        model_files = [f for f in os.listdir(self.model_dir) if f.startswith("dqn_model_")]

        if not model_files:
            print("No model found, training from scratch.")
            return

        latest_model = max(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        model_path = os.path.join(self.model_dir, latest_model)
        checkpoint = torch.load(model_path)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_net_state_dict'])
        self.epsilon = float(checkpoint('epsilon'))

    def store_data(self, state, action_lists, reward, next_state):

        # store all the elements
        self.replay_buffer.push(state, action_lists, reward, next_state)

        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

    def select_action(self, state_image, action_texts):
        """
        选择动作（ε-贪心策略）
        :param action_texts:
        :param state_image: 当前状态（图像）
        :return: 选择的动作索引
        """
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_actions)  # 随机选择动作
            r_values = torch.zeros((self.num_actions, 1)).to(
                self.device)  # Initialize a tensor of zeros with the size of num_actions
            r_values[action] = -0.01  # Update the corresponding index of r_values
            print("action:", action, "r_values:", r_values)
        else:
            with torch.no_grad():
                state_image = self.encoder_net.encode_image(state_image).float()
                text_actions = self.encoder_net.encode_text(action_texts).float()  # 获取所有动作的文本描述
                r_values = self.policy_net(state_image, text_actions)  # 计算奖励值r
                action = r_values.argmax().item()  # 选择 r 值最大的动作
                if action == 0:
                    r_values[action] = r_values[action] - 0.01
                print("action:", action, "r_values:", r_values)

        return action, action_texts[action], r_values  # 返回动作编号 + 对应的文本描述

    def train(self, batch_size, num_step):
        if len(self.replay_buffer) < batch_size:
            return

        # batch = self.replay_buffer.sample(batch_size)
        # states, actions_list, rewards, next_states = zip(*batch)
        #
        # states = torch.stack([self.encoder_net.encode_image(s) for s in states])
        # print(f"STATE shape: {states.shape}")
        # next_states = torch.stack([self.encoder_net.encode_image(s) for s in next_states])
        # # 编码动作文本特征（假设 action_list 是动作索引列表）
        # text_features = torch.stack([
        #     self.encoder_net.encode_text(action_idx)
        #     for action_idx in actions_list
        # ]).to(self.device)  # (batch_size, 512)
        # print(f"text stacked:{text_features.shape}")
        #
        # rewards = torch.stack(rewards).to(self.device)  # 确保 rewards 在正确的设备上
        # print(f"rewards stacked:{rewards.shape}")
        #
        # # 计算当前策略的 log π(a|s)
        # log_probs = F.log_softmax(self.policy_net(states, text_features), dim=-1)
        # # 计算每个动作的 log π(a|s) 选择性地通过 actions_list 的相对奖励进行加权
        # # 对于每个样本，我们获取所有动作的概率
        # action_log_probs = torch.gather(log_probs, 1, actions_list.unsqueeze(1))  # (batch_size, num_actions)
        #
        # # 计算旧策略（target net）的 log π_old(a|s)
        # with torch.no_grad():
        #     old_log_probs = F.log_softmax(self.target_net(states, text_features), dim=-1)  # (batch_size, num_actions)
        #     old_action_log_probs = torch.gather(old_log_probs, 1, actions_list.unsqueeze(1))  # (batch_size, num_actions)
        #
        # # 计算概率比率 r(θ) = π(a|s) / π_old(a|s)
        # ratio = torch.exp(action_log_probs - old_action_log_probs)
        #
        # # 计算优势函数 A(s, a)（标准化奖励作为近似优势）
        # advantages = rewards - rewards.mean(dim=-1, keepdim=True)  # (batch_size, num_actions)
        #
        # # GRPO 损失（裁剪形式）
        # clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        # policy_loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
        #
        # self.optimizer.zero_grad()
        # policy_loss.backward()
        # self.optimizer.step()
        #
        # # 更新 epsilon
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # tb_writer.add_scalar("Loss/train", policy_loss.item(), num_step)
        # tb_writer.add_scalar("state/train", states, num_step)
        # tb_writer.add_scalar("actions/train", actions_list, num_step)

        batch = self.replay_buffer.sample(batch_size)

        policy_losses = []
        all_states = []
        all_actions = []
        all_rewards = []
        for sample in batch:
            state, actions_list, reward, next_state = sample  # 拆分 batch 里的单个样本

            # 编码 state 和 next_state
            state = self.encoder_net.encode_image(state).float()  # (1, 512)
            # print("state is :", state)
            next_state = self.encoder_net.encode_image(next_state).float()  # (1, 512)

            # 编码动作文本特征
            text_features = self.encoder_net.encode_text(actions_list).float()

            # 确保 reward 维度匹配
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)  # (1, N)

            # 计算当前策略的 log π(a|s)
            log_probs = F.log_softmax(self.policy_net(state, text_features), dim=-1)  # (1, N)

            # 计算旧策略的 log π_old(a|s)
            with torch.no_grad():
                old_log_probs = F.log_softmax(self.target_net(state, text_features), dim=-1)  # (1, N)

            # 计算概率比率 r(θ) = π(a|s) / π_old(a|s)
            ratio = torch.exp(log_probs - old_log_probs)

            # 计算优势函数 A(s, a)
            advantages = reward - reward.mean(dim=-1, keepdim=True)  # (1, N)
            print()
            # GRPO 损失（裁剪形式）
            clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
            policy_loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))

            policy_losses.append(policy_loss)

        policy_loss = sum(policy_losses) / len(policy_losses)
        print("policy_loss is:", policy_losses)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # 更新 epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 记录训练数据
        tb_writer.add_scalar("Loss/train", policy_loss.item(), num_step)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def rebuild(self, state_image):
        with torch.no_grad():
            state_image_feature = self.encoder_net.encode_image(state_image).float()
            image_feature, att_weight, clustered_outputs, clusters = self.self_attention(state_image_feature)


