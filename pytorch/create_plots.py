from pathlib import Path
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

path = Path('/data/deephealth/deephealth-uc4/data/outputs/UNet2D/')

pretrained_iou1 = path / Path('pretrained/01-07_18-25/run-UNet2D_pretrained_01-07_18-25-tag-test_Mean_IoU.csv')
pretrained_iou2 = path / Path('pretrained/01-08_11-11/run-UNet2D_pretrained_01-08_11-11-tag-test_Mean_IoU.csv')
pretrained_iou3 = path / Path('pretrained/01-09_11-34/run-UNet2D_pretrained_01-09_11-34-tag-test_Mean_IoU.csv')

pretrained_dice1 = path / Path('pretrained/01-07_18-25/run-UNet2D_pretrained_01-07_18-25-tag-test_Mean_DiceCoeff.csv')
pretrained_dice2 = path / Path('pretrained/01-08_11-11/run-UNet2D_pretrained_01-08_11-11-tag-test_Mean_DiceCoeff.csv')
pretrained_dice3 = path / Path('pretrained/01-09_11-34/run-UNet2D_pretrained_01-09_11-34-tag-test_Mean_DiceCoeff.csv')

nopretrained_iou1 = path / Path('nopretrained/01-09_22-16/run-UNet2D_nopretrained_01-09_22-16-tag-test_Mean_IoU.csv')
nopretrained_iou2 = path / Path('nopretrained/01-10_10-00/run-UNet2D_nopretrained_01-10_10-00-tag-test_Mean_IoU.csv')
nopretrained_iou3 = path / Path('nopretrained/01-11_00-53/run-UNet2D_nopretrained_01-11_00-53-tag-test_Mean_IoU.csv')

nopretrained_dice1 = path / Path('nopretrained/01-09_22-16/run-UNet2D_nopretrained_01-09_22-16-tag-test_Mean_DiceCoeff.csv')
nopretrained_dice2 = path / Path('nopretrained/01-10_10-00/run-UNet2D_nopretrained_01-10_10-00-tag-test_Mean_DiceCoeff.csv')
nopretrained_dice3 = path / Path('nopretrained/01-11_00-53/run-UNet2D_nopretrained_01-11_00-53-tag-test_Mean_DiceCoeff.csv')

df_pretrained_iou1 = pd.read_csv(pretrained_iou1)[:100]
df_pretrained_iou2 = pd.read_csv(pretrained_iou2)[:100]
df_pretrained_iou3 = pd.read_csv(pretrained_iou3)[:100]
df_pretrained_dice1 = pd.read_csv(pretrained_dice1)[:100]
df_pretrained_dice2 = pd.read_csv(pretrained_dice2)[:100]
df_pretrained_dice3 = pd.read_csv(pretrained_dice3)[:100]

df_nopretrained_iou1 = pd.read_csv(nopretrained_iou1)[:100]
df_nopretrained_iou2 = pd.read_csv(nopretrained_iou2)[:100]
df_nopretrained_iou3 = pd.read_csv(nopretrained_iou3)[:100]
df_nopretrained_dice1 = pd.read_csv(nopretrained_dice1)[:100]
df_nopretrained_dice2 = pd.read_csv(nopretrained_dice2)[:100]
df_nopretrained_dice3 = pd.read_csv(nopretrained_dice3)[:100]

#df_nopretrained_dice1["Value"][21] = 0.6

pretrained_iou  = np.stack([df_pretrained_iou1["Value"],df_pretrained_iou2["Value"],df_pretrained_iou3["Value"]])
pretrained_iou_mean = pretrained_iou.mean(0)
pretrained_iou_std = pretrained_iou.std(0)

pretrained_dice  = np.stack([df_pretrained_dice1["Value"],df_pretrained_dice2["Value"],df_pretrained_dice3["Value"]])
pretrained_dice_mean = pretrained_dice.mean(0)
pretrained_dice_std = pretrained_dice.std(0)

nopretrained_iou  = np.stack([df_nopretrained_iou1["Value"],df_nopretrained_iou2["Value"],df_nopretrained_iou3["Value"]])
nopretrained_iou_mean = nopretrained_iou.mean(0)
nopretrained_iou_std = nopretrained_iou.std(0)

nopretrained_dice  = np.stack([df_nopretrained_dice1["Value"],df_nopretrained_dice2["Value"],df_nopretrained_dice3["Value"]])
nopretrained_dice_mean = nopretrained_dice.mean(0)
nopretrained_dice_std = nopretrained_dice.std(0)

plt.plot(pretrained_iou_mean, '--', color="#134f73", label='IoU score pretrained on LIDC')
plt.fill_between(range(100),pretrained_iou_mean-pretrained_iou_std,pretrained_iou_mean+pretrained_iou_std,alpha=.3,color="#134f73")
plt.plot(pretrained_dice_mean, color="#134f73", label='Dice score pretrained on LIDC')
plt.fill_between(range(100),pretrained_dice_mean-pretrained_dice_std,pretrained_dice_mean+pretrained_dice_std,alpha=.3,color="#134f73")

plt.plot(nopretrained_iou_mean, '--', color="#2fb0cd", label='IoU score trained from scratch')
plt.fill_between(range(100),nopretrained_iou_mean-nopretrained_iou_std,nopretrained_iou_mean+nopretrained_iou_std,alpha=.3,color="#2fb0cd")
plt.plot(nopretrained_dice_mean, color="#2fb0cd", label='Dice score trained from scratch')
plt.fill_between(range(100),nopretrained_dice_mean-nopretrained_dice_std,nopretrained_dice_mean+nopretrained_dice_std,alpha=.3,color="#2fb0cd")
#plt.ylim(0.2, 0.73)
plt.xlabel('Epoch')
plt.title('IoU and Dice scores for 2D U-Nets')
plt.legend()
plt.savefig("scores.pdf")
#print(len(df_pretrained_iou2))
#print(len(df_pretrained_iou3))

