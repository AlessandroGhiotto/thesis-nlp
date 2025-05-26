import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.lines as mlines

sns.set_style("darkgrid")


def basic_analysis(df, print_missing_values=True, print_count_statistics=True):
    """
    df should be a pandas DataFrame with the columns 'text' and 'label'

    This function will:
    - Display a WordCloud of the text corpus
    - Display a Pie Chart of the label distribution
    - Display a Histogram of the text length distribution
    - Print basic statistics about the dataset
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ---- 1️ WordCloud ----
    text_corpus = " ".join(str(text) for text in df["text"])
    wordcloud = WordCloud(
        width=800, height=800, background_color="white", colormap="viridis"
    ).generate(text_corpus)
    axes[0].imshow(wordcloud, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("WordCloud of Dataset")

    # ---- 2️ Pie Chart for Label Distribution ----
    label_counts = df["label"].value_counts()

    # Reduce label size if too many labels
    if len(label_counts) >= 10:
        label_fontsize = 8
    else:
        label_fontsize = 12
    axes[1].pie(
        label_counts,
        labels=label_counts.index,
        autopct="%1.1f%%",
        colors=plt.cm.Paired.colors,
        textprops={"fontsize": label_fontsize},
    )
    axes[1].set_title("Label Distribution")

    # ---- 3️ Text Length Distribution ----
    df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))  # Count words
    # Filter the dataset (just for visualization purposes)
    threshold = df["text_length"].quantile(0.995)  # Keep 99.5% of data
    filtered_df = df[df["text_length"] <= threshold]

    mean_length = df["text_length"].mean()

    sns.histplot(
        filtered_df["text_length"], bins=30, kde=False, ax=axes[2], label="Text Lengths"
    )
    axes[2].axvline(
        mean_length, color="red", linestyle="--", label=f"Mean = {mean_length:.2f}"
    )
    axes[2].set_title("Text Length Distribution")
    axes[2].set_xlabel("Number of words")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # ---- Print basic statistics ----
    print(f"Number of train samples: {len(df)}")
    print(f"Set of labels: {set(df['label'])}")
    print(f"Number of labels: {len(set(df['label']))}")
    if print_missing_values:
        print(f"Missing Values: \n{df.isnull().sum()}", sep="")
    if print_count_statistics:
        print("\nWord Count Statistics:")
        print(df["text_length"].describe().round(2))

    return None


def count_word_in_text(df, word, print_n=0):
    """
    Counts the percentage of documents containing 'word' in the 'text' field of the dataframe.
    """
    number_of_documents = (
        df["text"]
        .str.contains(r"\b" + re.escape(word) + r"\b", case=False, na=False)
        .sum()
    )
    percentage = (number_of_documents / len(df)) * 100 if len(df) > 0 else 0
    print(f"Percentage of documents containing '{word}': {percentage:.2f}%")
    if print_n > 0 and number_of_documents > 0:
        print("Examples:")
        for text in df[
            df["text"].str.contains(
                r"\b" + re.escape(word) + r"\b", case=False, na=False
            )
        ]["text"][:print_n]:
            print("-", text)
    return percentage


def starts_with_A(df, print_n=0):
    """
    Counts the percentage of documents starting with 'A ' in the 'text' field of the dataframe.
    """
    number_of_documents = df["text"].str.startswith("A ", na=False).sum()
    percentage = (number_of_documents / len(df)) * 100 if len(df) > 0 else 0
    print(f"Percentage of documents starting with 'A ': {percentage:.2f}%")
    if print_n > 0 and number_of_documents > 0:
        print("Examples:")
        for text in df[df["text"].str.startswith("A ", na=False)]["text"][:print_n]:
            print("-", text)
    return percentage


# from langdetect import detect, DetectorFactory
# DetectorFactory.seed = 0
# def count_non_english_articles(df, print_n=0):
#     """
#     Uses langdetect to find non-English articles in the 'text' column.
#     """
#     def is_non_english(text):
#         try:
#             return detect(text) != 'en'
#         except:
#             return True  # if detection fails, treat as non-English

#     non_english_mask = df['text'].apply(is_non_english)
#     count = non_english_mask.sum()
#     percentage = (count / len(df)) * 100 if len(df) > 0 else 0
#     print(f"Percentage of documents detected as non-English: {percentage:.2f}%")

#     if print_n > 0 and count > 0:
#         print("Examples:")
#         for text in df[non_english_mask]['text'][:print_n]:
#             print("-", text)

#     return percentage

# Precompiled regex for non-Latin scripts
non_latin_regex = re.compile(
    r"[\u4e00-\u9fff"  # CJK Unified Ideographs
    r"\u3040-\u309f"  # Hiragana
    r"\u30a0-\u30ff"  # Katakana
    r"\uac00-\ud7af"  # Hangul
    r"\u0400-\u04FF"  # Cyrillic
    r"\u0600-\u06FF"  # Arabic
    r"]"
)


def count_documents_with_non_latin(df, print_n=0):
    """
    Detects documents containing non-Latin script characters in 'text'.
    """
    mask = df["text"].str.contains(non_latin_regex, na=False)
    count = mask.sum()
    percentage = (count / len(df)) * 100 if len(df) > 0 else 0
    print(f"Documents containing non-Latin characters: {percentage:.2f}%")

    if print_n > 0 and count > 0:
        print("Examples:")
        for text in df[mask]["text"][:print_n]:
            print("-", text)

    return percentage


def count_world_in_non_world_label(df, print_n=0):
    """
    Counts the percentage of documents containing 'world' in the 'text' field where label is not 'World'.
    """
    mask = df["text"].str.contains(r"\bworld\b", case=False, na=False) & (
        df["label"] != "World"
    )
    count = mask.sum()
    percentage = (count / len(df)) * 100 if len(df) > 0 else 0
    print(
        f"Percentage of documents containing 'world' (but with label != 'World'): {percentage:.2f}%"
    )
    if print_n > 0 and count > 0:
        print("Examples:")
        texts = df[mask]["text"][:print_n]
        labels = df[mask]["label"].values[:print_n]
        for text, label in zip(texts, labels):
            print("-", text, " (label:", label, ")")
    return percentage


def check_patterns(df, print_n=0, dataset_name=None):
    """
    Check for specific patterns in the text field of the dataframe.
    Returns a dictionary with the results.
    """
    results = {}
    if dataset_name:
        results["dataset_name"] = dataset_name
    results["n_documents"] = len(df)

    # average length of text
    results["avg_length"] = df["text"].apply(lambda x: len(str(x).split())).mean()

    # Percentage of documents starting with 'A'
    results["starts_with_A_"] = starts_with_A(df, print_n)
    print()

    # Percentage of occurrences of 'World Cup' in the text field
    results["world_cup"] = count_word_in_text(df, "World Cup", print_n)
    print()

    # Percentage of occurrences of 'AI' in the text field
    results["ai"] = count_word_in_text(df, "AI", print_n)
    print()

    # Percentage of occurrences of 'world' in non-World labeled documents
    results["world_in_non_world_label"] = count_world_in_non_world_label(df, print_n)
    print()

    # Percentage of non-English articles (non-Latin characters)
    results["non_latin"] = count_documents_with_non_latin(df, print_n)

    return results


def plot_f1_lineplot(df, x_col="synthetic_ratio", hue_col="generation_method"):
    df_long = df.melt(
        id_vars=[x_col, hue_col],
        value_vars=["micro-f1", "macro-f1"],
        var_name="metric",
        value_name="score",
    )

    # Build relplot
    g = sns.relplot(
        data=df_long,
        x=x_col,
        y="score",
        col="metric",
        hue=hue_col,
        style=hue_col,
        kind="line",
        markers=True,
        dashes=False,
        height=6,
        aspect=1.2,
        facet_kws={"sharey": True},
    )

    # Adjust marker size and line width
    for ax in g.axes.flat:
        for line in ax.lines:
            line.set_markersize(12)  # marker size
            line.set_linewidth(2)  # line width

    g.set_titles(col_template="{col_name}")

    # for ax in g.axes.flat:
    #     ax.set_ylim(0, 1.05)

    plt.show()


def plot_f1_generationMethod(df, x_col="synthetic_ratio", hue_col="generation_method"):
    df_long = df.melt(
        id_vars=[x_col, hue_col],
        value_vars=["micro-f1", "macro-f1"],
        var_name="metric",
        value_name="score",
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    cmap = sns.color_palette("tab10")
    palette = {
        #'real': 'red', # cmap[3]
        "generic": cmap[0],
        "targeted": cmap[1],
        "unsupContext": cmap[2],
        "zeroshotLabels": cmap[4],
    }
    markers = {
        #'real': '*',
        "generic": "X",
        "targeted": "s",
        "unsupContext": "P",
        "zeroshotLabels": "D",
    }

    solid_dashes = {k: "" for k in df_long[hue_col].unique()}
    for ax, metric in zip(axes, df_long["metric"].unique()):
        # Exclude 'real' from the plot
        df_metric = df_long[
            (df_long["metric"] == metric) & (df_long[hue_col] != "real")
        ]

        sns.lineplot(
            data=df_metric,
            x=x_col,
            y="score",
            hue=hue_col,
            style=hue_col,
            dashes=solid_dashes,
            markers=markers,
            palette=palette,
            ax=ax,
            linewidth=2,
            markersize=10,
            legend=False,
        )

        # Add horizontal real-only line
        real_score = df_long[
            (df_long[hue_col] == "real") & (df_long["metric"] == metric)
        ]["score"].values[0]
        ax.axhline(
            y=real_score, linestyle="--", color=cmap[3], linewidth=2, alpha=1, zorder=0
        )

        ax.set_title(metric)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel("Score")

    custom_lines = [
        mlines.Line2D(
            [],
            [],
            color=palette[k],
            marker=markers[k],
            linestyle="-",
            markersize=10,
            label=k,
        )
        for k in df_long[hue_col].unique()
        if k != "real"
    ]
    # custom_lines.append(
    #     mlines.Line2D([], [], color='red', marker='*', linestyle='None', markersize=15, label='real')
    # )
    custom_lines.append(
        mlines.Line2D(
            [], [], color=cmap[3], linestyle="--", linewidth=2, label=r"100% real"
        )
    )

    fig.legend(
        handles=custom_lines,
        title=hue_col.replace("_", " ").title(),
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 0.92, 1])

    # fig.legend(
    #     handles=custom_lines,
    #     title="Generation Method",
    #     loc='lower center',
    #     bbox_to_anchor=(0.5, -0.06),
    #     ncol=3,
    #     frameon=False
    # )
    # plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.show()
