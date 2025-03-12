# config.py

class Config:
    # 訓練 CSV 路徑
    TRAIN_CSV = r"C:\ixi\IXI MNI\trainset\train_labels.csv"

    # 訓練影像資料夾
    GLOBAL_IMG_DIR = r"C:\ixi\IXI MNI\trainset\global"
    LOCAL_IMG_DIR  = r"C:\ixi\IXI MNI\trainset\local"

    # 測試 CSV 路徑
    TEST_CSV = r"C:\ixi\IXI MNI\testset\test_labels.csv"

    # 測試影像資料夾
    TEST_GLOBAL_DIR = r"C:\ixi\IXI MNI\testset\global"
    TEST_LOCAL_DIR  = r"C:\ixi\IXI MNI\testset\local"

    # 訓練參數
    EPOCHS        = 300
    BATCH_SIZE    = 4
    LEARNING_RATE = 1e-4
    VAL_RATIO     = 0.15

    # 模型儲存路徑
    SAVE_MODEL_PATH = r"C:\ixi\IXI MNI\best_model.pth"
