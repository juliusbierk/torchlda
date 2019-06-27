import torch

def lda(x1, x2, device="cpu"):
    with torch.no_grad():
        x1 = torch.tensor(x1, device=device, dtype=torch.float)
        x2 = torch.tensor(x2, device=device, dtype=torch.float)

        m1 = torch.mean(x1, dim=0)
        m2 = torch.mean(x2, dim=0)
        m = (len(x1) * m1 + len(x2) * m2) / (len(x1) + len(x2))

        d1 = x1 - m1[None, :]
        scatter1 = d1.t() @ d1
        d2 = x2 - m2[None, :]
        scatter2 = d2.t() @ d2
        within_class_scatter = scatter1 + scatter2

        d1 = m1 - m[None, :]
        scatter1 = len(x1) * (d1.t() @ d1)
        d2 = m2 - m[None, :]
        scatter2 = len(x2) * (d2.t() @ d2)
        between_class_scatter = scatter1 + scatter2

        p = torch.pinverse(within_class_scatter) @ between_class_scatter
        eigenvalues, eigenvectors = torch.eig(p, eigenvectors=True)
        idx = torch.argsort(eigenvalues[:, 0], descending=True)
        eigenvalues = eigenvalues[idx, 0]
        eigenvectors = eigenvectors[idx, :]

        return eigenvectors[0, :].cpu().numpy()
