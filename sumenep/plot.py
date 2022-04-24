import matplotlib.pyplot as plt


def show_probabilities_plot(labels, values, title):
    # Shot Probability Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylim((0, 1.2))
    ax.set_yticks([])

    ## Barplot
    cmap = plt.get_cmap("tab10")
    barplot = ax.bar(x=labels, height=values, width=0.4, color=cmap(values))

    ## Annotations
    rects = barplot.patches
    fontdict = dict(size=12)
    for rect in rects:
        value = f"{rect.get_height():.2%}"
        ax.text(x=rect.get_x() + 0.1, y=rect.get_height() + 0.05, s=value, fontdict=fontdict)

    return fig