from torch import lgamma
import torch.nn.functional as F
import torch


def variational_encoder_decoder(layers, input_data):
    l1, l2, l3, l4, alpha, l3_ ,l2_, l1_, rec = layers
    # We give batch_size x original_features_dim
    out = F.tanh(l1(input_data))
    out = F.tanh(l2(out))
    out = F.tanh(l3(out))
    ####### Variational part
    out = alpha(out)
    alpha_positive = F.sigmoid(out)
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