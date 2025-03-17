import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionWithClustering(nn.Module):
    def __init__(self, embed_dim, num_heads, threshold=0.5):
        super(SelfAttentionWithClustering, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.threshold = threshold  # 相关度阈值

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        clusters = self._cluster_attention_scores(attn_weights)
        clustered_outputs = self._generate_clustered_outputs(x, clusters, attn_weights)
        return attn_output, attn_weights, clustered_outputs, clusters

    def _cluster_attention_scores(self, attn_weights):
        batch_size, seq_len, seq_len2 = attn_weights.shape
        clusters = []

        for b in range(batch_size):
            visited = set()
            batch_clusters = []

            for i in range(seq_len2):
                if i not in visited:
                    high_attn_indices = (attn_weights[b, i] > self.threshold).nonzero(as_tuple=True)[0].tolist()
                    batch_clusters.append(high_attn_indices)
                    visited.update(high_attn_indices)

            clusters.append(batch_clusters)
        return clusters

    def _generate_image_features(self, x, clusters):
        batch_size, seq_len, embed_dim = x.shape
        image_features = torch.full_like(x, fill_value=-1e9)  # 用极小值填充

        for b, batch_clusters in enumerate(clusters):
            for cluster in batch_clusters:
                cluster_indices = torch.tensor(cluster, device=x.device)
                image_features[b, cluster_indices] = x[b, cluster_indices]  # 保留聚类内 token，其他部分保持极小值

        return image_features