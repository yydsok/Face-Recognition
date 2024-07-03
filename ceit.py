from transformers import ViTModel, ViTConfig
import torch.nn as nn

class CustomViTModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomViTModel, self).__init__()
        # 加载预训练模型和配置
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # 添加一个线性分类层
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, images):
        # 提取Transformer的输出
        outputs = self.vit(images)
        # 使用pooler_output进行分类（假设已经进行了pooling操作）
        x = outputs.pooler_output
        # 如果pooler_output是None，使用last_hidden_state的第一个元素
        if x is None:
            x = outputs.last_hidden_state[:, 0]  # 取序列的第一个元素，通常是CLS标记
        logits = self.classifier(x)
        return logits