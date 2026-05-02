import torch
import torch.nn as nn
import torch.nn.functional as F

class GoitaNet(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=65):
        """
        AIの脳の構造（ネットワーク）を定義します。
        
        :param input_size: 入力層のサイズ（盤面状態の特徴量の数）
        :param hidden_size: 隠れ層のニューロン数（AIの思考の複雑さ・表現力）
        :param output_size: 出力層のサイズ（AIが選択できる全アクションの数）
        """
        super(GoitaNet, self).__init__()
        
        # 第1層（入力層 -> 隠れ層1）
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 第2層（隠れ層1 -> 隠れ層2）
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 第3層（隠れ層2 -> 出力層）
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # 過学習（達人の棋譜の丸暗記による応用力低下）を防ぐためのドロップアウト
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        データがネットワークをどのように流れるか（順伝播）を定義します。
        """
        # 入力データを第1層に通し、ReLU（活性化関数）で非線形なパターンを抽出
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 第2層を通す（より深い文脈の理解）
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # 最終的な各アクションのスコア（Logits）を出力
        x = self.fc3(x)
        
        return x

# 動作確認用モック
if __name__ == "__main__":
    # 仮の入力サイズ（手札8種 + 場に見えている駒8種 + 現在の攻め駒など = 例: 30）
    mock_input_size = 30
    # 出力サイズ（例: パス1種 + (伏せ8種 × 攻め8種) = 最大65パターンの行動）
    mock_output_size = 65
    
    # 脳みそ（モデル）の実体化
    model = GoitaNet(input_size=mock_input_size, output_size=mock_output_size)
    print("モデルの構造:")
    print(model)