from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    })
test_loss_csv = "eddl/wandb/run_2s0msy84_test_loss.csv"
training_loss_csv = "eddl/wandb/run_2s0msy84_training_loss.csv"
validation_loss_csv = "eddl/wandb/run_2s0msy84_validation_loss.csv"

test_loss = pd.read_csv(test_loss_csv)
training_loss = pd.read_csv(training_loss_csv)
validation_loss = pd.read_csv(validation_loss_csv)

print(test_loss["swept-water-137 - test-loss"])


plt.plot(training_loss["Step"], training_loss["swept-water-137 - train-loss"], label = "Training", color="#4b61d1", linestyle="solid")
plt.plot(training_loss["Step"], validation_loss["swept-water-137 - validation-loss"], label = "Validation", color="#4b61d1", linestyle="dashed")
plt.plot(training_loss["Step"], test_loss["swept-water-137 - test-loss"], label = "Test", color="#4b61d1", linestyle="dotted")
plt.ylabel("Dice loss")
plt.xlabel("Epoch")
plt.title("Dice loss for U-Net model with PyECVL and PyEDDL")
plt.legend(facecolor="white")
plt.savefig("dice_loss.pdf")

