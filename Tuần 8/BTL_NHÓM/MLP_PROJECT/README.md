# BÀI TẬP XẤP XỈ HÀM SỐ DÙNG MẠNG MLP

## 1. Giới thiệu
Dự án sử dụng mạng nơ-ron nhân tạo MLP (Multi-Layer Perceptron) để xấp xỉ một hàm Gaussian hai chiều.

- Input: hai biến đầu vào x1 và x2.
- Output: giá trị hàm Gaussian tương ứng.
- Framework: PyTorch.
- Loss function: Mean Squared Error (MSE).
- Optimizer: Adam.

## 2. Cấu trúc thư mục

```
MLP_PROJECT/
│
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── fig1_true_surface.png
│   └── fig3_distributions.png
│
├── main_code/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│   └── visualize_chapter5.py

│
├── experiments/
│   ├── experiment_A.py
│   ├── experiment_B.py
│   ├── experiment_C.py
│   └── config_A.json
│
├── results/
│   ├── csv/
│   │   ├── loss_history_A.csv
│   │   ├── loss_history_B.csv
│   │   ├── loss_history_C.csv
│   │   ├── results_A_runs.csv
│   │   ├── results_A.csv
│   │   ├── results_B_raw.csv
│   │   ├── results_B.csv
│   │   ├── results_C_raw.csv
│   │   └── results_C.csv
│   │
│   └── figures/
│       ├── chapter5_fig_5_1_true_vs_pred.png
│       ├── chapter5_fig_5_2_absolute_error.png
│       ├── fig_A1_error_vs_samples.png
│       ├── fig_A2_train_val_test_mse.png
│       ├── fig_A3_learning_curves.png
│       ├── fig_A4_pred_vs_true_surface.png
│       ├── plot_B_errors.png
│       ├── plot_B_loss_curves.png
│       ├── plot_B_neurons.png
│       ├── plot_B_training_time.png
│       ├── plot_C_errors.png
│       ├── plot_C_layers.png
│       ├── plot_C_loss_curves.png
│       └── plot_C_training_time.png
│       
│
├── checkpoints/
│
├── requirements.txt
└── README.md     
```

## 3. Cài đặt môi trường

Yêu cầu Python 3.10 trở lên.

Cài đặt các thư viện bằng lệnh:

```bash
python --version
python -m pip install -r requirements.txt
```

## 4. Chạy chương trình

Mở Terminal tại thư mục MLP_PROJECT

### Chạy chương trình chính

```bash
cd "Desktop\BT Học Sâu\MLP_PROJECT"
```

### Thực nghiệm A – Ảnh hưởng của kích thước dữ liệu

```bash
python experiments/experiment_A.py
```

### Thực nghiệm B – Ảnh hưởng của số neuron

```bash
python experiments/experiment_B.py
```

### Thực nghiệm C – Ảnh hưởng của số tầng ẩn

```bash
python experiments/experiment_C.py
```

## 5. Kết quả đầu ra

Sau khi thực hiện các chương trình, kết quả được lưu trong thư mục results/, bao gồm các tệp CSV và biểu đồ trực quan.

-`results/csv/`: Lưu kết quả huấn luyện, lịch sử loss và kết quả đánh giá các thực nghiệm

-`figures/`: Lưu các biểu đồ thể hiện kết quả thực nghiệm.

-`checkpoints/`: Lưu trọng số hoặc mô hình đã huấn luyện.

## 6. Đánh giá mô hình

Mô hình được đánh giá bằng các chỉ số:

- MSE (Sai số bình phương trung bình)
- RMSE (Căn bậc hai của sai số bình phương trung bình)
- MAE (Sai số tuyệt đối trung bình)

## 7. Các thực nghiệm

- Experiment A: Khảo sát ảnh hưởng của số mẫu huấn luyện (A) đến hiệu năng mạng MLP.
- Experiment B: Khảo sát ảnh hưởng của độ rộng mạng MLP đến khả năng xấp xỉ hàm Gaussian hai chiều. Mạng có đúng một tầng ẩn, số nơ-ron khảo sát từ 1 đến 10.
- Experiment C: Khảo sát ảnh hưởng của độ sâu mạng MLP đến sai số xấp xỉ hàm Gaussian hai chiều, khả năng hội tụ và thời gian huấn luyện. Số tầng ẩn khảo sát từ 1 đến 4.
