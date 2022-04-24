import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
import plotly.express as px

from utils import get_group_count


def show_probabilities_plot(labels, values, title):
    # Setup figure
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


def show_pie_chart(counts):
    fig = px.pie(values=counts.values, 
                 names=counts.index, 
                 width=900, 
                 height=720,)

    fig.update_traces(
        textposition ='inside', 
        textinfo='percent + label', 
        hole=0.5, 
        marker=dict(colors = ['navy','red','orange'], line=dict(color='white', width=3))
    )

    fig.update_layout(
        title_text='IKS Sumenep', 
        title_x=0.5, 
        title_y=0.53, 
        title_font_size=32, 
        title_font_family='Calibri Black', 
        title_font_color='black', 
        showlegend=False)           

    return fig


def show_multiclass_plot(df, x, y, hue):
    # Setup figure
    fig, ax = plt.subplots()

    # Barplot
    a = sns.barplot(data=df, x=x, y=y, hue=hue, palette=['red', 'navy','orange'], ax=ax)
    ax.set_title(x, size=14)
    ax.set(xlabel=None, ylabel=None)
    ax.legend(loc="upper right")

    # Annotations
    for p in a.patches:
        height = p.get_height()
        a.annotate(f'{height:g}', (p.get_x() + p.get_width() / 2, p.get_height()), 
                    ha='center', va='center', 
                    size=10,
                    xytext=(0, 5), 
                    textcoords='offset points')
    
    return fig


def make_figure(df, refs, target):
    sns.set_style('white')
    figs = []

    for ref in refs:
        group_count_df = get_group_count(df, ref, target)
        fig = show_multiclass_plot(df=group_count_df, x=ref, y="jumlah", hue=target)
        figs.append(fig)

    return figs