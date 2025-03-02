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
    l1, l2, l3, l4, alpha_layers, l3_ ,l2_, l1_, rec = layers
    # We give batch_size x original_features_dim
    out = F.tanh(l1(input_data))
    out = F.tanh(l2(out))
    out = F.tanh(l3(out))
    ####### Variational part
    concentration_params = []
    K = len(alpha_layers)
    for k in range(K):
        scale = (k + 1) * 10
        layer = alpha_layers[k]
        a = layer(out)  # scale * (1 + F.relu(layer(h1)))  # this makes each k (group) in the prod with different level of values
        concentration_params.append(a)
    size = list(concentration_params[0].size())
    x, y = size[0], size[1]
    em_normalized = torch.zeros(x, y, K)
    alphas = torch.stack(concentration_params)  # 10 * (1 + F.relu(torch.stack(concentration_params)))

    alphas_pos = []

    for j in range(x):
        for k in range(K):
            alpha_positive = F.sigmoid(alphas[k, j, :])
            alphas_pos.append(alpha_positive)
            u = torch.rand(alpha_positive.shape)
            temp = u.mul(alpha_positive).mul(torch.exp(lgamma(alpha_positive)))
            em = torch.pow(temp, torch.pow(alpha_positive, -1))
            em_normalized[j, :, k] = F.softmax(em)

    em_final = torch.flatten(em_normalized, start_dim=1)
    alphas = torch.stack(alphas_pos)

    #######
    # We get batch_size x embedding_dim
    out = F.tanh(l3_(em_final))
    out = F.tanh(l2_(out))
    if l1_.in_features == out.shape[1]:
        out = F.tanh(l1_(out))
    # We get batch_size x original_features_dim
    reconstructed = torch.sigmoid(rec(out))
    return em_final, alphas.view(K, x, y), reconstructed