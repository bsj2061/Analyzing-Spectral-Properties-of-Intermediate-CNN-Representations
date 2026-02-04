import torch
from src.data import transforms

class MasterSpectralAnalyzer:
    def __init__(self, device):
        self.device = device

    def get_rank(self, output, target_idx):
        if isinstance(target_idx, torch.Tensor):
            if target_idx.dim() == 0:
                t_idx = target_idx.item()
            else:
                t_idx = target_idx[0].item()
        else:
            t_idx = target_idx

        logits = output[0] if output.dim() > 1 else output

        sorted_indices = torch.argsort(logits, dim=0, descending=True)
        rank = (sorted_indices == t_idx).nonzero(as_tuple=True)[0].item() + 1
        return rank

    def run_experiment(self, model, loader, layers, bands, num_images=10):

        results = {l: {b: [] for b in self.bands} for l in layers}

        count = 0
        for imgs, labels in loader:
            if count >= num_images: break
            imgs = imgs.to(self.device)
            # labels는 get_rank에서 안전하게 처리됨

            for l_name in layer:
                for b_name, (r_min, r_max) in bands.items():
                    transform = transforms.FourierBandTransform(r_min, r_max)
                    model.eval()
                    with torch.no_grad():
                        output = model.preprocess(x)
                        if model_name == 'resnet18':
                            x = model.maxpool(model.relu(model.bn1(model.conv1(imgs))))
                            layers = [model.layer1, model.layer2, model.layer3, model.layer4]
                            for i, layer in enumerate(layers):
                                x = layer(x)
                                if l_name == f'layer{i+1}':
                                    x = transform(x)
                            x = model.avgpool(x)
                            output = model.fc(torch.flatten(x, 1))
                        else:
                            x = model._process_input(imgs)
                            n_batch = x.shape[0]
                            x = torch.cat([model.class_token.expand(n_batch, -1, -1), x], dim=1)
                            x = x + model.encoder.pos_embedding
                            for i, layer in enumerate(model.encoder.layers):
                                x = layer(x)
                                if l_name == f'encoder.layers.{i}':
                                    cls_t, patch_t = x[:, :1, :], x[:, 1:, :]
                                    B, N, C = patch_t.shape
                                    H = W = int(N**0.5)
                                    spatial_p = patch_t.transpose(1, 2).reshape(B, C, H, W)
                                    filtered_p = transform(spatial_p)
                                    patch_t = filtered_p.reshape(B, C, N).transpose(1, 2)
                                    x = torch.cat([cls_t, patch_t], dim=1)
                            output = model.heads(model.encoder.ln(x)[:, 0])

                        results[l_name][b_name].append(self.get_rank(output, labels))

            count += 1
            print(f"📊 Image {count}/{num_images} processed...", end="\r")

        # 결과 출력
        #log(results)
        #plots(results)
        header = f"{'Layer':<18}" + "".join([f"| {b:<15}" for b in self.bands.keys()])
        print(f"\n\n{header}")
        print("-" * len(header))
        for l_name in layer_names:
            row = f"{l_name:<18}"
            for b_name in self.bands:
                mean_rank = np.mean(results[l_name][b_name])
                row += f" | {mean_rank:<13.2f}"
            print(row)

# --- 실행부 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
analyzer = MasterSpectralAnalyzer(device)

# ResNet 분석
analyzer.run_experiment('resnet18', loader, num_images=50)

# ViT 분석
analyzer.run_experiment('vit_b_16', loader, num_images=50)

