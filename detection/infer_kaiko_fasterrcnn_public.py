import os
import glob
import json
import torch
from torchvision import transforms
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import datetime  # 日時用
import numpy as np  # 移動平均用

# 学習時と同じモデル定義を再現する関数
def get_model(num_classes):
    # torchvision の学習済み Faster R-CNN をロード
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # background=0, kaiko_live=1, kaiko_pao1=2 の3クラスに対応
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# 推論を行い、PIL.Image と1画像分の推論結果（辞書）を返す関数
def run_inference(image_path, model, device):
    # 画像の読み込みと前処理
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model([img_tensor])  # outputs はリストなので最初のものを採用
    return img, outputs[0]

# 検出結果を可視化する関数（画像保存にも利用可能）
def visualize_detection(image, output, score_threshold=0.7, save_path=None):
    """
    image: PIL.Image形式の画像
    output: 推論結果（辞書形式。 "boxes", "labels", "scores"が含まれる）
    score_threshold: 信頼度がこの値以上の検出のみを描画
    save_path: 保存する場合はファイルパスを指定
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    
    # マッピング例：0は背景（描画しない）
    # 1 を "kaiko_live"（青）、2 を "kaiko_pao1"（赤）として表示
    label_names = {1: "kaiko_live", 2: "kaiko_pao1"}
    label_colors = {1: "blue", 2: "red"}
    
    for box, score, label in zip(boxes, scores, labels):
        if label == 0 or score < score_threshold:
            continue
        
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        
        color = label_colors.get(label, "black")
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2,
                                 edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        caption = f"{label_names.get(label, str(label))}: {score:.2f}"
        ax.text(xmin, ymin - 5, caption, fontsize=12, color="yellow",
                backgroundcolor=color)
    
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# 簡易的な移動平均関数
def moving_average(data, window_size=3):
    # data を numpy 配列にして、'same'モードで畳み込みを行う
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

if __name__ == '__main__':
    # 設定：クラス数、デバイスなど
    num_classes = 3  # background, kaiko_live, kaiko_pao1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 公開版のため、ローカル絶対パスはプレースホルダ化しています。
    checkpoint_path = "PATH/TO/fasterrcnn_kaiko_2025-0219.pth"
    image_folder = "PATH/TO/INFERENCE_IMAGES"
    if checkpoint_path.startswith("PATH/TO/") or image_folder.startswith("PATH/TO/"):
        raise ValueError("プレースホルダのままです。checkpoint_path と image_folder を実パスに置き換えてください。")

    # モデル作成と重みの読み込み
    model = get_model(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    # 複数画像の格納フォルダ（jpgファイルを対象）
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    
    # 各画像ごとの検出結果（タイムコース）を保存するリスト
    detection_results = []
    # 推論結果画像の保存先フォルダ（存在しなければ作成）
    vis_save_dir = os.path.join(image_folder, "vis_results")
    os.makedirs(vis_save_dir, exist_ok=True)
    
    # 各画像について推論を実施し、各クラスの検出件数をカウント
    for idx, image_path in enumerate(image_paths):
        print(f"Processing [{idx+1}/{len(image_paths)}]: {os.path.basename(image_path)}")
        
        # ファイル名から日時情報を抽出（例："YYYYMMDD_HHMMSS.jpg"）
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # datetimeオブジェクトに変換
        timestamp = datetime.datetime.strptime(base_name, "%Y%m%d_%H%M%S")
        
        original_img, detection_output = run_inference(image_path, model, device)
        
        # 閾値以上の検出のみカウント
        score_thr = 0.7
        boxes = detection_output["boxes"].cpu().numpy()
        scores = detection_output["scores"].cpu().numpy()
        labels = detection_output["labels"].cpu().numpy()
        
        count_live = 0
        count_pao1 = 0
        for box, score, label in zip(boxes, scores, labels):
            if score < score_thr or label == 0:
                continue
            if label == 1:
                count_live += 1
            elif label == 2:
                count_pao1 += 1
        
        # JSONに保存できる形へ変換
        boxes_list = boxes.tolist()    # 各ボックスは [xmin, ymin, xmax, ymax] の形式
        scores_list = scores.tolist()
        labels_list = labels.tolist()
        
        # 検出結果にタイムスタンプも保存
        detection_results.append({
            "image": os.path.basename(image_path),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "count_kaiko_live": count_live,
            "count_kaiko_pao1": count_pao1,
            "boxes": boxes_list,
            "scores": scores_list,
            "labels": labels_list
            })
        
        # オプション：検出結果の画像（visualize_detection 内で枠描画）
        vis_path = os.path.join(vis_save_dir, os.path.basename(image_path))
        visualize_detection(original_img, detection_output, score_threshold=score_thr, save_path=vis_path)
    
    # 結果を JSON ファイルとして保存
    results_file = os.path.join(image_folder, "detection_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detection_results, f, indent=2, ensure_ascii=False)
    print(f"検出結果は {results_file} に保存されました。")
    
    # タイムコースグラフ用時系列データの生成
    timestamps = []
    for res in detection_results:
        dt = datetime.datetime.strptime(res["timestamp"], "%Y-%m-%d %H:%M:%S")
        timestamps.append(dt)
    
    t0 = timestamps[0]
    # 各画像の最初の時刻からの経過時間（hours）を算出 (3600秒 = 1時間)
    elapsed_hours = [ (t - t0).total_seconds() / 3600.0 for t in timestamps ]
    
    live_counts = [res["count_kaiko_live"] for res in detection_results]
    pao1_counts = [res["count_kaiko_pao1"] for res in detection_results]
    
    # 比率 (kaiko_live / (kaiko_live + kaiko_pao1)) を計算　※合計が0の場合は 0 とする
    ratio = []
    for live, pao1 in zip(live_counts, pao1_counts):
        if (live + pao1) > 0:
            ratio.append(live / (live + pao1))
        else:
            ratio.append(0)
    
    # 比率のスムージング（移動平均）; window_size はデータ数に応じて調整してください
    smoothed_ratio = moving_average(ratio, window_size=3)
    
    # グラフ描画(サブプロットで2パネルを作成)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # ① 上段(ax1)に kaiko_live / kaiko_pao1 の検出件数プロット
    ax1.plot(elapsed_hours, live_counts, marker='o', label="kaiko_live", color='blue')
    ax1.plot(elapsed_hours, pao1_counts, marker='o', label="kaiko_pao1", color='red')
    ax1.set_ylabel("Detection Count")
    ax1.set_title("Time Course of Detections")
    ax1.legend()
    ax1.grid(True)
    
    # ② 下段(ax2)に比率の生データとスムージング曲線のプロット
    ax2.scatter(elapsed_hours, ratio, label="Raw Ratio", color='purple')
    ax2.plot(elapsed_hours, smoothed_ratio, label="Smoothed Ratio", color='green', linestyle="--")
    ax2.set_xlabel("Elapsed Time (hours)")
    ax2.set_ylabel("Ratio [kaiko_live/(kaiko_live+pao1)]")
    ax2.set_title("Time Course of Ratio")
    ax2.set_ylim(0, 1)     # 縦軸範囲を0～1に設定
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # グラフの保存
    graph_save_path = os.path.join(image_folder, "timecourse_plot.png")
    plt.savefig(graph_save_path, bbox_inches="tight")
    plt.show()
    print(f"タイムコースのグラフは {graph_save_path} に保存されました。")
