import argparse
parser = argparse.ArgumentParser(description="FIDTM")


parser.add_argument(
    "--dataset",
    type=str,
    default="ShanghaiA",  #修改数据集
    choices=[
        "ShanghaiA",  #1
        "ShanghaiB",
        "UCF_QNRF",
        "UCF50_1",  #2
        "UCF50_2", #3
        "UCF50_3",#4
        "UCF50_4",#5
        "UCF50_5",#6
        "JHU",
        "CARPK",   #7
        "PUCPR",
        "large",
        "small",
        "ship",
        "building",
        "TRANCOS",
        'NWPU',
    ],
    help="choice train dataset",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="/scratch/chen_quantum/shanghaiA_5xr_test",  #最优模型的保存地址
    help="cho`ice train dataset",
)
parser.add_argument("--workers", type=int, default=16, help="load data workers=8 or 16")
parser.add_argument("--print_freq", type=int, default=200, help="print frequency")
parser.add_argument(
    "--start_epoch", type=int, default=0, help="start epoch for training"
 )
parser.add_argument("--epochs", type=int, default=3000, help="end epoch for training")
parser.add_argument("--pre", type=str, default=None, help="pre-trained model directory")
parser.add_argument(
    "--batch_size", type=int, default=64, help="input batch size for training"   #default=16
)

parser.add_argument(
    "--crop_size",
    type=int,
    default=256,
    choices=[256, 512],
    help="crop size for training, SHHA/SHHB:256, QNRF/JHU/UCF50:512",
)
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--best_pred", type=int, default=1e5, help="best pred")
parser.add_argument("--best_mse", type=int, default=1e5, help="best mse")
parser.add_argument("--gpu_id", type=str, default="0,1", help="gpu id")    ##修改gpu id
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=5*1e-4, help="weight decay")#5 * 1e-4
parser.add_argument("--preload_data", type=bool, default=True, help="preload data. ")
parser.add_argument(
    "--visual", type=bool, default=False, help="visual for bounding box. "
)
parser.add_argument("--video_path", type=str, default=None, help="input video path ")
parser.add_argument("--quantum", action="store_true", default=False)
parser.add_argument("--transformer", action="store_true", default=False)
parser.add_argument("--qmblk", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--heatmap", action="store_true", default=False)
parser.add_argument("--del_seed", action="store_false")
parser.add_argument("--loss", type=str, default="MSE", choices=["MSE", "CALoss"])
parser.add_argument("--network", type=str, default="os")   ####修改网络
parser.add_argument("--dimension", type=int, default=512)
parser.add_argument("--number", type=int, default=2)
parser.add_argument("--depth", type=int, default=1)
parser.add_argument("--code", type=str, default=None)


args = parser.parse_args()
return_args = parser.parse_args()
