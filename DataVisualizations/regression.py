# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from pandas.plotting import parallel_coordinates
# import numpy as np

# # =======================
# # Load Data
# # =======================
# file_path = "TreasureHunt.csv"  # Adjust file path as needed
# df = pd.read_csv(file_path)

# # =======================
# # Setup Plot Style
# # =======================
# sns.set(style="whitegrid")

# # =======================
# # Select Questions for Analysis
# # =======================
# questions = df.columns[1:-1]  # Skip timestamp/free-text columns (assumed first and last)
# selected_questions = questions[:7]

# # =======================
# # Encode Categorical Columns
# # =======================
# encoded_df = df[selected_questions].apply(lambda col: col.astype('category').cat.codes)

# # =======================
# # 1. Correlation Heatmap
# # =======================
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     encoded_df.corr(),
#     annot=True,
#     cmap="coolwarm",
#     fmt=".2f",
#     linewidths=0.5,
#     square=True
# )
# plt.title("Correlation Between Survey Responses")
# plt.tight_layout()
# plt.show()

# # =======================
# # 2. Scatterplot: Question 1 vs Question 2
# # =======================
# plt.figure(figsize=(6, 4))
# sns.scatterplot(
#     x=encoded_df[selected_questions[0]],
#     y=encoded_df[selected_questions[1]],
#     hue=df[selected_questions[2]],  # Use original categorical data for coloring
#     palette="viridis"
# )
# plt.title(f"Comparison: {selected_questions[0]} vs {selected_questions[1]}")
# plt.xlabel(selected_questions[0])
# plt.ylabel(selected_questions[1])
# plt.tight_layout()
# plt.show()

# # =======================
# # 3. Simple Linear Regression: Predict Q1 from Q2
# # =======================
# X = encoded_df[[selected_questions[1]]]
# y = encoded_df[selected_questions[0]]

# model = LinearRegression()
# model.fit(X, y)
# pred = model.predict(X)

# plt.figure(figsize=(6, 4))
# plt.scatter(X, y, alpha=0.7, label="Actual")
# plt.plot(X, pred, color="red", linewidth=2, label="Regression Line")
# plt.xlabel(selected_questions[1])
# plt.ylabel(selected_questions[0])
# plt.title("Linear Regression: Predict Q1 from Q2")
# plt.legend()
# plt.tight_layout()
# plt.show()

# print(f"Simple Regression R² score: {r2_score(y, pred):.3f}")

# # =======================
# # 4. Multiple Linear Regression: Predict Q1 from other Questions
# # =======================
# X_multi = encoded_df[selected_questions[1:]]
# y_multi = encoded_df[selected_questions[0]]

# multi_model = LinearRegression()
# multi_model.fit(X_multi, y_multi)
# multi_pred = multi_model.predict(X_multi)

# print(f"Multiple Regression R² score: {r2_score(y_multi, multi_pred):.3f}")
# print("Coefficients:")
# for question, coef in zip(selected_questions[1:], multi_model.coef_):
#     print(f"  {question}: {coef:.3f}")
# print(f"Intercept: {multi_model.intercept_:.3f}")

# # =======================
# # 5. Pairplot with Truncated Labels & Better Layout
# # =======================
# def truncate_label(label, max_len=10):
#     label = str(label)
#     if len(label) > max_len:
#         return label[:max_len-3] + "..."
#     return label

# subset_questions = selected_questions[:4]
# subset_encoded_df = encoded_df[subset_questions]

# # Truncate column names for readability
# truncated_columns = [truncate_label(col) for col in subset_questions]
# subset_encoded_df.columns = truncated_columns

# g = sns.pairplot(
#     subset_encoded_df,
#     corner=True,
#     plot_kws={'alpha': 0.5, 's': 30},
#     diag_kind='kde',
#     height=3.5,
#     aspect=1
# )

# # Rotate x-axis labels on bottom row
# for ax in g.axes[-1, :]:
#     for label in ax.get_xticklabels():
#         label.set_rotation(45)
#         label.set_horizontalalignment('right')

# # Rotate y-axis labels on first column
# for ax in g.axes[:, 0]:
#     for label in ax.get_yticklabels():
#         label.set_rotation(45)
#         label.set_verticalalignment('center')

# plt.suptitle("Pairwise Comparisons (Truncated Labels)", y=1.02)
# plt.tight_layout()
# plt.show()

# # =======================
# # Additional Visualizations
# # =======================

# # 6. Boxplots for Each Question
# plt.figure(figsize=(10,6))
# sns.boxplot(data=encoded_df, palette="Set2")
# plt.title("Boxplot of Encoded Survey Responses")
# plt.xlabel("Questions")
# plt.ylabel("Encoded Response Values")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # 7. Violin Plots for Distribution by Question
# plt.figure(figsize=(10,6))
# sns.violinplot(data=encoded_df, palette="Set3")
# plt.title("Violin Plot of Survey Responses")
# plt.xlabel("Questions")
# plt.ylabel("Encoded Response Values")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # 8. Countplot for Each Question (Categorical Frequencies)
# fig, axes = plt.subplots(3, 3, figsize=(15,12))
# axes = axes.flatten()
# for i, question in enumerate(selected_questions):
#     sns.countplot(x=df[question], ax=axes[i], palette="pastel")
#     axes[i].set_title(f"Response Counts: {question}")
#     axes[i].tick_params(axis='x', rotation=45)
# # Hide unused subplots if any
# for j in range(i+1, len(axes)):
#     fig.delaxes(axes[j])
# plt.tight_layout()
# plt.show()

# # 9. Heatmap of Missing Values
# plt.figure(figsize=(8,4))
# sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
# plt.title("Missing Values Heatmap")
# plt.tight_layout()
# plt.show()

# # 10. Pairwise Correlation Scatterplot with Regression Line (Q2 predicting Q1)
# plt.figure(figsize=(8,6))
# sns.regplot(x=encoded_df[selected_questions[1]], y=encoded_df[selected_questions[0]], scatter_kws={'alpha':0.5})
# plt.title(f"Regression Plot: {selected_questions[1]} predicting {selected_questions[0]}")
# plt.xlabel(selected_questions[1])
# plt.ylabel(selected_questions[0])
# plt.tight_layout()
# plt.show()

# # 11. Bar Plot of Mean Responses per Question
# mean_responses = encoded_df.mean()
# plt.figure(figsize=(10,5))
# sns.barplot(x=mean_responses.index, y=mean_responses.values, palette="Blues_d")
# plt.title("Average Encoded Response per Question")
# plt.ylabel("Mean Response")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # 12. Stacked Bar Chart of Response Distribution
# response_counts = {}
# for q in selected_questions:
#     response_counts[q] = df[q].value_counts(normalize=True).sort_index()
# response_df = pd.DataFrame(response_counts).fillna(0).T

# response_df.plot(
#     kind='bar',
#     stacked=True,
#     figsize=(12,7),
#     colormap='tab20'
# )
# plt.title("Stacked Bar Chart of Response Distribution")
# plt.xlabel("Questions")
# plt.ylabel("Proportion of Responses")
# plt.xticks(rotation=45)
# plt.legend(title='Response Category', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# # 13. Radar Chart (Spider Plot) for Average Responses
# labels = selected_questions[:5]
# stats = encoded_df[labels].mean().values

# angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
# stats = np.concatenate((stats,[stats[0]]))
# angles += angles[:1]

# fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
# ax.plot(angles, stats, 'o-', linewidth=2)
# ax.fill(angles, stats, alpha=0.25)
# ax.set_thetagrids(np.degrees(angles[:-1]), labels)
# ax.set_title("Radar Chart of Average Encoded Responses")
# ax.grid(True)
# plt.show()

# # 14. Parallel Coordinates Plot
# subset_questions_pc = selected_questions[:5]
# subset_encoded_df_pc = encoded_df[subset_questions_pc].copy()
# subset_encoded_df_pc['Group'] = df[selected_questions[2]].astype(str)  # Use Q3 as group

# plt.figure(figsize=(12,6))
# parallel_coordinates(subset_encoded_df_pc, 'Group', colormap=plt.get_cmap("Set2"))
# plt.title("Parallel Coordinates Plot Grouped by Question 3")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # 15. Clustered Heatmap
# sns.clustermap(encoded_df, cmap="coolwarm", standard_scale=1, figsize=(10,10))
# plt.title("Clustered Heatmap of Encoded Responses")
# plt.show()

# # 16. Line Plot of Responses Over Questions (Sample of 5 Respondents)
# plt.figure(figsize=(10,5))
# for i in range(min(5, len(encoded_df))):
#     plt.plot(selected_questions, encoded_df.iloc[i, :], marker='o', label=f"Respondent {i+1}")

# plt.title("Response Trends Across Questions (Sample of 5 Respondents)")
# plt.xlabel("Questions")
# plt.ylabel("Encoded Response")
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # 17. Distribution Histograms per Question
# fig, axes = plt.subplots(3, 3, figsize=(15,12))
# axes = axes.flatten()
# for i, question in enumerate(selected_questions):
#     axes[i].hist(encoded_df[question], bins=10, color='skyblue', edgecolor='black')
#     axes[i].set_title(f"Histogram: {question}")
#     axes[i].set_xlabel("Encoded Response")
#     axes[i].set_ylabel("Count")
# # Hide unused subplots if any
# for j in range(i+1, len(axes)):
#     fig.delaxes(axes[j])
# plt.tight_layout()
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pandas.plotting import parallel_coordinates
import numpy as np

# =======================
# Helper to truncate labels
# =======================
def truncate_label(label, max_len=7):
    label = str(label)
    if len(label) > max_len:
        return label[:max_len-3] + "..."
    return label

# =======================
# Load Data
# =======================
file_path = "TreasureHunt.csv"  # Adjust file path as needed
df = pd.read_csv(file_path)

# =======================
# Setup Plot Style
# =======================
sns.set(style="whitegrid")

# =======================
# Select Questions for Analysis
# =======================
questions = df.columns[1:-1]  # Skip timestamp/free-text columns (assumed first and last)
selected_questions = questions[:7]

# Apply truncation to selected questions for display
trunc_questions = [truncate_label(q) for q in selected_questions]

# =======================
# Encode Categorical Columns
# =======================
encoded_df = df[selected_questions].apply(lambda col: col.astype('category').cat.codes)

# =======================
# 1. Correlation Heatmap
# =======================
plt.figure(figsize=(8, 6))
sns.heatmap(
    encoded_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    square=True,
    xticklabels=trunc_questions,
    yticklabels=trunc_questions
)
plt.title("Correlation Between Survey Responses")
plt.tight_layout()
plt.show()

# =======================
# 2. Scatterplot: Question 1 vs Question 2
# =======================
plt.figure(figsize=(6, 4))
sns.scatterplot(
    x=encoded_df[selected_questions[0]],
    y=encoded_df[selected_questions[1]],
    hue=df[selected_questions[2]],  # Use original categorical data for coloring
    palette="viridis"
)
plt.title(f"Comparison: {truncate_label(selected_questions[0])} vs {truncate_label(selected_questions[1])}")
plt.xlabel(truncate_label(selected_questions[0]))
plt.ylabel(truncate_label(selected_questions[1]))
plt.tight_layout()
plt.show()

# =======================
# 3. Simple Linear Regression: Predict Q1 from Q2
# =======================
X = encoded_df[[selected_questions[1]]]
y = encoded_df[selected_questions[0]]

model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

plt.figure(figsize=(6, 4))
plt.scatter(X, y, alpha=0.7, label="Actual")
plt.plot(X, pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel(truncate_label(selected_questions[1]))
plt.ylabel(truncate_label(selected_questions[0]))
plt.title("Linear Regression: Predict Q1 from Q2")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Simple Regression R² score: {r2_score(y, pred):.3f}")

# =======================
# 4. Multiple Linear Regression: Predict Q1 from other Questions
# =======================
X_multi = encoded_df[selected_questions[1:]]
y_multi = encoded_df[selected_questions[0]]

multi_model = LinearRegression()
multi_model.fit(X_multi, y_multi)
multi_pred = multi_model.predict(X_multi)

print(f"Multiple Regression R² score: {r2_score(y_multi, multi_pred):.3f}")
print("Coefficients:")
for question, coef in zip(selected_questions[1:], multi_model.coef_):
    print(f"  {truncate_label(question)}: {coef:.3f}")
print(f"Intercept: {multi_model.intercept_:.3f}")

# =======================
# 5. Pairplot with Truncated Labels & Better Layout
# =======================
subset_questions = selected_questions[:4]
subset_encoded_df = encoded_df[subset_questions].copy()

# Create a new DataFrame with truncated column names
truncated_columns = [truncate_label(col) for col in subset_questions]
trunc_df = subset_encoded_df.copy()
trunc_df.columns = truncated_columns

g = sns.pairplot(
    trunc_df,
    corner=True,
    plot_kws={'alpha': 0.5, 's': 30},
    diag_kind='kde',
    height=3.5,
    aspect=1
)

# Rotate x-axis labels on bottom row
for ax in g.axes[-1, :]:
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')

# Rotate y-axis labels on first column
for ax in g.axes[:, 0]:
    for label in ax.get_yticklabels():
        label.set_rotation(45)
        label.set_verticalalignment('center')

plt.suptitle("Pairwise Comparisons (Truncated Labels)", y=1.02)
plt.tight_layout()
plt.show()

# =======================
# Additional Visualizations
# =======================

# 6. Boxplots for Each Question
plt.figure(figsize=(10,6))
sns.boxplot(data=encoded_df, palette="Set2")
plt.title("Boxplot of Encoded Survey Responses")
plt.xlabel("Questions")
plt.ylabel("Encoded Response Values")
plt.xticks(ticks=range(len(trunc_questions)), labels=trunc_questions, rotation=45)
plt.tight_layout()
plt.show()

# 7. Violin Plots for Distribution by Question
plt.figure(figsize=(10,6))
sns.violinplot(data=encoded_df, palette="Set3")
plt.title("Violin Plot of Survey Responses")
plt.xlabel("Questions")
plt.ylabel("Encoded Response Values")
plt.xticks(ticks=range(len(trunc_questions)), labels=trunc_questions, rotation=45)
plt.tight_layout()
plt.show()

# 8. Countplot for Each Question (Categorical Frequencies)
fig, axes = plt.subplots(3, 3, figsize=(15,12))
axes = axes.flatten()
for i, question in enumerate(selected_questions):
    sns.countplot(x=df[question], ax=axes[i], palette="pastel")
    axes[i].set_title(f"Response Counts: {truncate_label(question)}")
    axes[i].tick_params(axis='x', rotation=45)
# Hide unused subplots if any
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# 9. Heatmap of Missing Values
plt.figure(figsize=(8,4))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.show()

# 10. Pairwise Correlation Scatterplot with Regression Line (Q2 predicting Q1)
plt.figure(figsize=(8,6))
sns.regplot(x=encoded_df[selected_questions[1]], y=encoded_df[selected_questions[0]], scatter_kws={'alpha':0.5})
plt.title(f"Regression Plot: {truncate_label(selected_questions[1])} predicting {truncate_label(selected_questions[0])}")
plt.xlabel(truncate_label(selected_questions[1]))
plt.ylabel(truncate_label(selected_questions[0]))
plt.tight_layout()
plt.show()

# 11. Bar Plot of Mean Responses per Question
mean_responses = encoded_df.mean()
plt.figure(figsize=(10,5))
sns.barplot(x=trunc_questions, y=mean_responses.values, palette="Blues_d")
plt.title("Average Encoded Response per Question")
plt.ylabel("Mean Response")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 12. Stacked Bar Chart of Response Distribution
response_counts = {}
for q in selected_questions:
    response_counts[q] = df[q].value_counts(normalize=True).sort_index()
response_df = pd.DataFrame(response_counts).fillna(0).T
response_df.index = [truncate_label(idx) for idx in response_df.index]

response_df.plot(
    kind='bar',
    stacked=True,
    figsize=(12,7),
    colormap='tab20'
)
plt.title("Stacked Bar Chart of Response Distribution")
plt.xlabel("Questions")
plt.ylabel("Proportion of Responses")
plt.xticks(rotation=45)
plt.legend(title='Response Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 13. Radar Chart (Spider Plot) for Average Responses
labels = selected_questions[:5]
trunc_labels = [truncate_label(l) for l in labels]
stats = encoded_df[labels].mean().values

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
stats = np.concatenate((stats,[stats[0]]))
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.plot(angles, stats, 'o-', linewidth=2)
ax.fill(angles, stats, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), trunc_labels)
ax.set_title("Radar Chart of Average Encoded Responses")
ax.grid(True)
plt.show()

# 14. Parallel Coordinates Plot
subset_questions_pc = selected_questions[:5]
subset_encoded_df_pc = encoded_df[subset_questions_pc].copy()
subset_encoded_df_pc['Group'] = df[selected_questions[2]].astype(str)  # Use Q3 as group

plt.figure(figsize=(12,6))
parallel_coordinates(subset_encoded_df_pc, 'Group', colormap=plt.get_cmap("Set2"))
plt.title("Parallel Coordinates Plot Grouped by Question 3")
plt.xticks(ticks=range(len(subset_questions_pc)), labels=[truncate_label(q) for q in subset_questions_pc], rotation=45)
plt.tight_layout()
plt.show()

# 15. Clustered Heatmap
sns.clustermap(encoded_df, cmap="coolwarm", standard_scale=1, figsize=(10,10))
plt.title("Clustered Heatmap of Encoded Responses")
plt.show()

# 16. Line Plot of Responses Over Questions (Sample of 5 Respondents)
plt.figure(figsize=(10,5))
for i in range(min(5, len(encoded_df))):
    plt.plot(trunc_questions, encoded_df.iloc[i, :], marker='o', label=f"Resp {i+1}")

plt.title("Response Trends Across Questions (Sample of 5 Respondents)")
plt.xlabel("Questions")
plt.ylabel("Encoded Response")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 17. Distribution Histograms per Question
fig, axes = plt.subplots(3, 3, figsize=(15,12))
axes = axes.flatten()
for i, question in enumerate(selected_questions):
    axes[i].hist(encoded_df[question], bins=10, color='skyblue', edgecolor='black')
    axes[i].set_title(f"Histogram: {truncate_label(question)}")
    axes[i].set_xlabel("Encoded Resp")
    axes[i].set_ylabel("Count")
# Hide unused subplots if any
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()
