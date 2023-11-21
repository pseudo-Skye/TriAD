import torch
import torch.nn.functional as F

def ts_loss(z1, z2, alpha=0.4):
    loss = alpha * cross_domain_loss(z1) + (1-alpha) * intra_domain_loss(z1, z2)
    return loss 

# z1 : D x B x T
def cross_domain_loss(z1):
    z1 = F.normalize(z1, p=2, dim=2)
    D, B = z1.size(0), z1.size(1)
    # 1. Find positive pairs
    sim = torch.abs(torch.matmul(z1, z1.transpose(1, 2))) # D x B x B
    sim_updated = torch.tril(sim, diagonal=-1)[:, :, :-1]    # D x B x (B-1)
    sim_updated += torch.triu(sim, diagonal=1)[:, :, 1:] # D x B x (B-1)
    pos = torch.exp(sim_updated).sum(dim=-1).unsqueeze(-1)  # D x B x 1

    # 2. Find negative pairs
    z = z1.transpose(0, 1)  # B x D x T
    sim = torch.abs(torch.matmul(z, z.transpose(1, 2))) # B x D x D
    sim_updated = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x D x (D-1)
    sim_updated += torch.triu(sim, diagonal=1)[:, :, 1:] # B x D x (D-1)
    neg = torch.exp(sim_updated).sum(dim=-1)  # B x D
    neg = neg.transpose(0, 1).unsqueeze(-1) # D x B x 1

    # 3. Cat the pos and neg pairs
    # Concate the postives and negatives
    logits = torch.cat([pos, neg], dim = -1) # D x B x 2
    logits = -torch.log(logits[:,:,0:1]/(logits.sum(dim=-1).unsqueeze(-1)))
    loss = logits.max()
    return loss

# z1 : D x B x T
def intra_domain_loss(z1, z2):
    z1 = F.normalize(z1, p=2, dim=2)
    z2 = F.normalize(z2, p=2, dim=2)
    D, B = z1.size(0), z1.size(1) 
    z = torch.cat([z1, z2], dim=1)  # D x 2B x T
    sim = torch.abs(torch.matmul(z, z.transpose(1, 2)))  # D x 2B x 2B
    # Remove the similarity between instance itself
    sim_updated = torch.tril(sim, diagonal=-1)[:, :, :-1]    # D x 2B x (2B-1)
    sim_updated += torch.triu(sim, diagonal=1)[:, :, 1:] 
    
    pos = torch.exp(sim_updated[:,0:B, 0:B-1]).sum(dim=-1).unsqueeze(-1) # D x B x 1
    neg = torch.exp((sim_updated[:, 0:B, B-1:] + sim_updated[:, B:, 0:B])/2).sum(dim=-1).unsqueeze(-1) # D x B x 1
    # Concate the postives and negatives
    logits = torch.cat([pos, neg], dim = -1) # D x B x 2
    logits = -torch.log(logits[:,:,0:1]/(logits.sum(dim=-1).unsqueeze(-1)))
    loss = logits.mean()
    return loss