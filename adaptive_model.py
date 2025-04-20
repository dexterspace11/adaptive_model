
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class AdaptiveParameterFolder:
    def __init__(self, compression_ratio=0.1):
        self.compression_ratio = compression_ratio
        self.parameter_folds = {}

    def fold_parameters(self, model):
        """Compress parameters while maintaining key features"""
        all_params = self.flatten_params(model)
        importance_scores = self.calculate_importance(all_params)
        threshold = np.percentile(importance_scores, 100 * (1 - self.compression_ratio))
        mask = importance_scores >= threshold
        compressed_params = all_params[mask]
        metadata = self.extract_metadata(model)
        return {'compressed': compressed_params, 'metadata': metadata}

    def flatten_params(self, model):
        """Flatten all model parameters into a single 1D numpy array"""
        return torch.cat([p.view(-1) for p in model.parameters()]).detach().cpu().numpy()

    def calculate_importance(self, params):
        """Simulate importance score calculation (e.g., magnitude-based)"""
        return np.abs(params)

    def extract_metadata(self, model):
        """Save shape of each layer to preserve structure"""
        return {name: param.shape for name, param in model.named_parameters()}


class EfficiencyAwareLearning:
    def __init__(self, target_loss=0.5, target_usage=100.0):
        self.learning_rate = 0.001
        self.target_loss = target_loss
        self.target_usage = target_usage
        self.efficiency_metrics = []

    def adapt_learning(self, loss, resource_usage):
        """Adjust learning behavior (placeholder for dynamic logic)"""
        if resource_usage > self.target_usage:
            self.adjust_complexity()
        elif loss > self.target_loss:
            self.increase_precision()

    def adjust_complexity(self):
        print("🔧 Reducing model complexity due to high resource usage.")

    def increase_precision(self):
        print("📈 Increasing precision due to high loss.")


class AdaptiveNeuralNet(nn.Module):
    def __init__(self, input_size):
        super(AdaptiveNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.out(x))
        return x


def train_model(model, data, targets, epochs=30, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    return model


def generate_sample_data(n_samples=100, input_dim=10):
    X = np.random.rand(n_samples, input_dim)
    y = (np.sum(X, axis=1) > (input_dim / 2)).astype(float)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def compress_and_report(model):
    folder = AdaptiveParameterFolder(compression_ratio=0.1)
    result = folder.fold_parameters(model)
    print("✅ Compression Complete")
    print("Compressed Parameter Count:", len(result['compressed']))
    print("Model Metadata:", result['metadata'])
    return result


# If run as script, train a sample model and compress it
if __name__ == "__main__":
    input_size = 10
    X, y = generate_sample_data(input_dim=input_size)
    model = AdaptiveNeuralNet(input_size)
    trained_model = train_model(model, X, y)
    compress_and_report(trained_model)
