from torch import lgamma
import torch.nn.functional as F
import torch


def reduce_correlation(v1, v2):
    # Ensure the vectors are tensors
    v1 = torch.tensor(v1, dtype=torch.float32)
    v2 = torch.tensor(v2, dtype=torch.float32)

    # Step 1: Project v2 onto v1
    projection = (torch.dot(v2, v1) / torch.dot(v1, v1)) * v1

    # Step 2: Subtract the projection from v2 to reduce correlation
    v2_orthogonal = v2 - projection

    # Optionally: Normalize both vectors if necessary (comment out if not needed)
    v1_normalized = v1 / v1.norm()
    v2_orthogonal_normalized = v2_orthogonal / v2_orthogonal.norm()

    v1_normalized[v1_normalized < 0] = 1
    v2_orthogonal_normalized[v2_orthogonal_normalized < 0] = 1

    return v1_normalized, v2_orthogonal_normalized


def variational_encoder_decoder(layers, input_data):
    l1, l2, l3, l4, alpha, l3_ ,l2_, l1_, rec = layers
    # We give batch_size x original_features_dim
    out = F.tanh(l1(input_data))
    out = F.tanh(l2(out))
    out = F.tanh(l3(out))
    ####### Variational part
    out = alpha(out)
    alpha_positive = F.sigmoid(out)  # torch.abs(F.tanh(alpha)) + 0.0001 # 1 + F.relu(alpha)  # F.sigmoid(alpha)
    u = torch.rand(alpha_positive.shape)
    temp = u.mul(alpha_positive).mul(torch.exp(lgamma(alpha_positive)))
    em = torch.pow(temp, torch.pow(alpha_positive, -1))
    em_normalized = F.softmax(em, dim=1)
    #######
    # We get batch_size x embedding_dim
    out = F.tanh(l3_(em_normalized))
    out = F.tanh(l2_(out))
    if l1_.in_features == out.shape[1]:
        out = F.tanh(l1_(out))
    # We get batch_size x original_features_dim
    reconstructed = torch.sigmoid(rec(out))
    return em_normalized, alpha_positive, reconstructed