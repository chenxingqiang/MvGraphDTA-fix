import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MvGraphDTA import *
from evaluate_metrics import *
from utils import *
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def pad_features(features, target_dim):
    """将特征张量填充到目标维度"""
    current_dim = features.size(-1)
    if current_dim >= target_dim:
        return features

    # 计算需要填充的维度数
    pad_size = target_dim - current_dim

    # 创建填充张量（用0填充）
    padding = torch.zeros(
        *features.shape[:-1], pad_size, device=features.device)

    # 连接原始特征和填充
    padded_features = torch.cat([features, padding], dim=-1)

    return padded_features


def load_data(id, data_set='test'):
    path = './data/binding_affinity/PDBbindv2016/core_set/'

    # 加载drug节点特征
    drug_node_feature = torch.load(
        path + 'drug_node_feature/' + id[0] + '.pt').to(device)
    print("Original Drug Node Feature Shape:", drug_node_feature.shape)

    # 将drug_node_feature从27维扩展到29维
    drug_node_feature = pad_features(drug_node_feature, 29)
    print("Padded Drug Node Feature Shape:", drug_node_feature.shape)

    drug_edge_feature = torch.load(
        path + 'drug_edge_feature/' + id[0] + '.pt').to(device)
    print("Drug Edge Feature Shape:", drug_edge_feature.shape)

    drug_edge_index = torch.load(
        path + 'drug_edge_index/' + id[0] + '.pt').to(device)
    drug_line_edge_index = torch.load(
        path + 'drug_line_edge_index/' + id[0] + '.pt').to(device)
    drug_node_edge_index = torch.load(
        path + 'drug_node_edge_index/' + id[0] + '.pt').to(device)
    drug_edge_node_index = torch.load(
        path + 'drug_edge_node_index/' + id[0] + '.pt').to(device)
    drug_node_edge_scatter_index = torch.load(
        path + 'drug_node_edge_scatter_index/' + id[0] + '.pt').to(device)
    drug_edge_node_scatter_index = torch.load(
        path + 'drug_edge_node_scatter_index/' + id[0] + '.pt').to(device)

    target_node_feature = torch.load(
        path + 'target_node_feature/' + id[0] + '.pt').to(device)
    target_edge_feature = torch.load(
        path + 'target_edge_feature/' + id[0] + '.pt').to(device)

    print("Target Node Feature Shape:", target_node_feature.shape)
    print("Target Edge Feature Shape:", target_edge_feature.shape)

    target_edge_index = torch.load(
        path + 'target_edge_index/' + id[0] + '.pt').to(device)
    target_line_edge_index = torch.load(
        path + 'target_line_edge_index/' + id[0] + '.pt').to(device)
    target_node_edge_index = torch.load(
        path + 'target_node_edge_index/' + id[0] + '.pt').to(device)
    target_edge_node_index = torch.load(
        path + 'target_edge_node_index/' + id[0] + '.pt').to(device)
    target_node_edge_scatter_index = torch.load(
        path + 'target_node_edge_scatter_index/' + id[0] + '.pt').to(device)
    target_edge_node_scatter_index = torch.load(
        path + 'target_edge_node_scatter_index/' + id[0] + '.pt').to(device)

    drug_data = drug_node_feature, drug_edge_index, drug_edge_feature, drug_line_edge_index, drug_node_edge_index, drug_edge_node_index, drug_node_edge_scatter_index, drug_edge_node_scatter_index

    target_data = target_node_feature, target_edge_index, target_edge_feature, target_line_edge_index, target_node_edge_index, target_edge_node_index, target_node_edge_scatter_index, target_edge_node_scatter_index

    return drug_data, target_data


def validation(model, loader, epoch=1, epochs=1, data_set='test'):
    """验证/测试函数"""
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for batch, (id, affinity) in enumerate(loader):
            drug_data, target_data = load_data(id, data_set=data_set)
            output = model(drug_data, target_data)
            total_preds = torch.cat((total_preds, output.detach().cpu()), 0)
            total_labels = torch.cat(
                (total_labels, affinity.view(-1, 1).cpu()), 0)

    total_labels = total_labels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()
    return total_labels, total_preds


if __name__ == '__main__':
    device = torch.device('cpu')

    # 使用原始的29维，因为我们会将输入数据填充到这个维度
    drug_n_dim = 29
    drug_v_dim = 5
    target_n_dim = 20
    target_v_dim = 2
    emb_dim = 128
    out_dim = 256
    n_class = 1
    batch_size = 1
    num_layers = 2
    hidden_dim = [128, 128]

    test_data = dataset('./data/binding_affinity/PDBbindv2016/testing.csv')
    test_loader = DataLoader(
        test_data, batch_size=batch_size, num_workers=14, shuffle=False)

    model = PredicterDTA(drug_n_dim, drug_v_dim, target_n_dim, target_v_dim,
                         hidden_dim, out_dim, num_layers, n_class).to(device)
    model.eval()

    # 加载预训练模型
    model.load_state_dict(torch.load(
        './data/best_model/MvGraphDTA_Drug_Similarity.pt',
        map_location=torch.device('cpu'),
        weights_only=True  # 添加此参数以避免警告
    ))

    # 进行测试
    test_labels, test_preds = validation(model, test_loader, data_set='test')
    test_result = [
        mae(test_labels, test_preds),
        rmse(test_labels, test_preds),
        pearson(test_labels, test_preds),
        spearman(test_labels, test_preds),
        ci(test_labels, test_preds),
        r_squared(test_labels, test_preds)
    ]

    print("\nTest Results:")
    print(f"MAE: {test_result[0]:.4f}")
    print(f"RMSE: {test_result[1]:.4f}")
    print(f"Pearson: {test_result[2]:.4f}")
    print(f"Spearman: {test_result[3]:.4f}")
    print(f"CI: {test_result[4]:.4f}")
    print(f"R²: {test_result[5]:.4f}")
