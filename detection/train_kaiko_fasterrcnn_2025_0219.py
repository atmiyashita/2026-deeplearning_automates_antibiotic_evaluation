#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import json

# 複数のフォルダとアノテーションファイルに対応するデータセットクラス
class KaikoDataset(torch.utils.data.Dataset):
    def __init__(self, roots, annotation_files, transforms=None):
        """
        roots: 画像ファイルのディレクトリ、文字列または文字列のリスト
        annotation_files: 各画像のアノテーションが記載されたJSONファイルのパス、
                          文字列または文字列のリスト
        transforms: 画像に適用する変換（必要に応じて）
        """
        self.transforms = transforms
        self.datasets = []  # 各フォルダとJSONデータをまとめるリスト

        # rootsとannotation_filesがリストの場合、そのペアごとに処理
        if isinstance(roots, list) and isinstance(annotation_files, list):
            if len(roots) != len(annotation_files):
                raise ValueError("rootsとannotation_filesの数が一致している必要があります。")
            for root, ann_file in zip(roots, annotation_files):
                with open(ann_file, 'r', encoding='utf-8') as f:
                    annotations_json = json.load(f)
                asset_keys = list(annotations_json["assets"].keys())
                self.datasets.append({
                    "root": root,
                    "annotations_json": annotations_json,
                    "asset_keys": asset_keys
                })
        else:
            # 引数が文字列の場合はリストに変換して統一
            with open(annotation_files, 'r', encoding='utf-8') as f:
                annotations_json = json.load(f)
            asset_keys = list(annotations_json["assets"].keys())
            self.datasets.append({
                "root": roots,
                "annotations_json": annotations_json,
                "asset_keys": asset_keys
            })

        # 各データセットのサンプル数と、全体の累積サンプル数を計算
        self.cumulative_lengths = []
        total = 0
        for ds in self.datasets:
            total += len(ds["asset_keys"])
            self.cumulative_lengths.append(total)

        print("各データセットのサンプル数:", [len(ds["asset_keys"]) for ds in self.datasets])
        print("全体のサンプル数:", self.__len__())

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        # idx がどのデータセットに属するか調べ、ローカルインデックスを求める
        dataset_idx = 0
        for cum_len in self.cumulative_lengths:
            if idx < cum_len:
                break
            dataset_idx += 1
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        ds = self.datasets[dataset_idx]

        # JSON上のキーからアノテーション情報を取得
        asset_key = ds["asset_keys"][local_idx]
        asset_info = ds["annotations_json"]["assets"][asset_key]

        # asset_info["asset"]内にファイル名などの情報があると仮定
        file_name = asset_info["asset"]["name"]
        img_path = os.path.join(ds["root"], file_name)

        # 画像の読み込み
        img = Image.open(img_path).convert("RGB")
        image_width, image_height = img.size

        boxes = []
        labels = []
        # regions キーに各オブジェクトのアノテーションが格納されていると仮定
        regions = asset_info.get("regions", [])
        for region in regions:
            # region["boundingBox"] にバウンディングボックスの情報（left, top, width, height）があると仮定
            bb = region["boundingBox"]
            left = bb["left"]
            top  = bb["top"]
            width  = bb["width"]
            height = bb["height"]

            # バウンディングボックスのサイズが正であることを確認（ゼロや負の場合はスキップ）
            if width <= 0 or height <= 0:
                print(f"Warning: 無効な領域がスキップされました: {bb}")
                continue

            xmin = left
            ymin = top
            xmax = left + width
            ymax = top + height
            boxes.append([xmin, ymin, xmax, ymax])

            # アノテーションに含まれるラベル情報の取得とマッピング
            tags = region.get("tags", [])
            label_str = ""
            if tags:
                label_str = tags[0].strip().lower()   
            print("label_str=" + label_str)
            # "molt" を含むなら kaiko_pao1 (2)、"live" を含むなら kaiko_live (1)
            if "pao1" in label_str:
                labels.append(2)
            elif "live" in label_str:
                labels.append(1)
            else:
                print(f"Warning: 未対応または空のラベル '{label_str}' が使用されています。デフォルトで kaiko_live を適用。")
                labels.append(1)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img = self.transforms(img)

        return img, target

# 学習用モデルの作成（Faster R-CNNのカスタマイズ）
def get_model(num_classes):
    # torchvisionの学習済みFaster R-CNNをロード
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # background=0, kaiko_live=1, kaiko_pao1=2
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# メイン処理（シンプルな学習ループ例）
def main():
    # ハイパーパラメータなど
    num_classes = 3  # background, kaiko_live, kaiko_pao1
    num_epochs = 10
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 変換処理（Tensor変換だけ）
    from torchvision import transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # データセットの作成
    dataset = KaikoDataset(dataset_root, annotation_file, transforms=transform)

    # DataLoaderの作成
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )

    # モデルの作成
    model = get_model(num_classes)
    model.to(device)

    # オプティマイザの設定
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 学習ループ
    print("訓練開始")
    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} 損失: {losses.item():.4f}")

    # 学習済みモデルの保存
    torch.save(model.state_dict(), "fasterrcnn_kaiko_2025-0219.pth")
    print("学習終了・モデルを保存しました。")

if __name__ == '__main__':
    # dataset_root と annotation_file はリスト形式で指定
    # 公開版のため、ローカル絶対パスはプレースホルダ化しています。
    path1 = "PATH/TO/2025-0211-1/vott-json-export"
    path2 = "PATH/TO/2025-0211-2/vott-json-export"
    path3 = "PATH/TO/2025-0212-1/vott-json-export"
    path4 = "PATH/TO/2025-0213-1/vott-json-export"
    path5 = "PATH/TO/2025-0213-2/vott-json-export"
    path6 = "PATH/TO/2025-0214-1/vott-json-export"
    path7 = "PATH/TO/2025-0214-2/vott-json-export"
    path8 = "PATH/TO/2025-0219-1/vott-json-export"
    path9 = "PATH/TO/2025-0219-2/vott-json-export"
    
    dataset_root = [path1, path2, path3, path4, path5, path6, path7, path8, path9]
    annotation_file = [path1 + "/atmiyashita-export.json",
                       path2 + "/atmiyashita-export.json",
                       path3 + "/atmiyashita-export.json",
                       path4 + "/atmiyashita-export.json",
                       path5 + "/atmiyashita-export.json",
                       path6 + "/atmiyashita-export.json",
                       path7 + "/atmiyashita-export.json",
                       path8 + "/atmiyashita-export.json",
                       path9 + "/atmiyashita-export.json"]

    if any(str(p).startswith("PATH/TO/") for p in dataset_root):
        raise ValueError("プレースホルダのままです。path1..path9 を実データのパスに置き換えてください。")

    main()
