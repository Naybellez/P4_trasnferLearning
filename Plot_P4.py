
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

def plot_boxplots(all_df, acm, randomChanceLevel):
    # acm: accuracy measure
    order = ['8c3l','7c3l','4c3l','3c2l','2c2l']
    df_all_copy = all_df.copy()
    mapping = {'[452, 144]' : '452 X 144', '[226, 72]' : '226 X 72', '[113, 36]' : '113 X 36', '[57, 18]' : '57 X 18', '[29, 9]' : '29 X 9'}
    df_all_copy['resolution'] = df_all_copy['resolution'].map(mapping)

    g = sns.FacetGrid(df_all_copy, col="resolution", hue="model_name", height=4, aspect=1.5, col_wrap=3).map(sns.boxplot, "model_name", acm).add_legend()
    if acm == 'TESTAccBase':
        g.set_axis_labels("Model Name", "Base Accuracy")
        g.map(plt.axhline, y=randomChanceLevel, ls='--', c='lightcoral', label='Chance Level')
    elif acm == 'TESTAccPeakDist':
        g.set_axis_labels("Model Name", "Peak Dist")
    plt.show()


def plot_boxplot2(all_df, acm, ChanceL1=None, ChanceL2=None):
    order = ['8c3l','7c3l','6c3l','4c3l','3c2l','2c2l']
    df_all_copy = all_df.copy()
    mapping = {'[452, 144]' : '452 X 144', '[226, 72]' : '226 X 72', '[113, 36]' : '113 X 36', '[57, 18]' : '57 X 18', '[29, 9]' : '29 X 9'}
    df_all_copy['resolution'] = df_all_copy['resolution'].map(mapping)

    colours = sns.color_palette("colorblind", n_colors=len(order))
    colours2 = colours[::-1]

    g = sns.catplot(
    data=df_all_copy,
    x="model_name",
    y=acm,
    col="resolution",
    hue="model_name",
    palette=colours2,
    order=order,
    kind="box",
    height=4,
    aspect=1.5,
    col_wrap=3
    )

    if ChanceL1 and acm == 'TESTAccBase':
        g.set_axis_labels("Model Name", "Base Accuracy")
        g.map(plt.axhline, y=ChanceL1, ls='--', c='lightcoral', label='Always Guess C')
        g.map(plt.axhline, y=ChanceL2, ls='--', c='lightblue', label='47/360')
    elif acm == 'TESTAccPeakDist':
        g.set_axis_labels("Model Name", "Peak Dist")
    else:
        print("Youre trying to plot TESTBaseAcc but havn't included the random chance levels!!!")

    # Get the model colours from the FacetGrid
    model_handles = [
        Line2D(
            [0], [0],
            marker="s",
            linestyle="",
            markerfacecolor=colour,
            markeredgecolor="none",
            markersize=8,
            label=model
        )
        for model, colour in zip(order,colours)
    ]

    # Handles for the chance lines
    chance_handles = [
        Line2D(
            [0], [0],
            color="lightcoral",
            linestyle="--",
            label=f"Always Guess C ({ChanceL1}%)"
        ),
        Line2D(
            [0], [0],
            color="lightblue",
            linestyle=":",
            label=f"Chance level (47/360={ChanceL2}%)"
        )
    ]

    # Combine them
    if ChanceL1 and acm == 'TESTAccBase':
        handles = model_handles + chance_handles
    else:
        handles = model_handles

    # Add legend
    g.figure.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(0.89, 0.23),
        fontsize=14,
        ncol=1,#len(handles),
        #frameon=False
    )   
