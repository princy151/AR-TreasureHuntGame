import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# Load CSV file
# =======================
file_path = "TreasureHunt.csv"  # <-- change if needed
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Set style
sns.set(style="whitegrid")

# =======================
# Select Questions
# =======================
questions = df.columns[1:]  # skip Timestamp
selected_questions = questions[:7]  # pick first 7 survey questions

# =======================
# 1. Bar plot - Puzzle & Narrative Promotion
# =======================
plt.figure(figsize=(8,5))
sns.countplot(
    y=df[selected_questions[0]],
    order=df[selected_questions[0]].value_counts().index,
    palette="viridis"
)
plt.title("Puzzles & Narratives Promotion")
plt.tight_layout()
plt.show()

# =======================
# 2. Pie chart - Preferred AR Approach
# =======================
plt.figure(figsize=(6,6))
df[selected_questions[1]].value_counts().plot.pie(
    autopct='%1.1f%%',
    colors=sns.color_palette("coolwarm", len(df[selected_questions[1]].unique()))
)
plt.title("Preferred AR Approach")
plt.ylabel("")
plt.tight_layout()
plt.show()

# =======================
# 3. Horizontal bar plot - Cross-platform Challenges
# =======================
plt.figure(figsize=(8,5))
sns.countplot(
    y=df[selected_questions[2]],
    order=df[selected_questions[2]].value_counts().index,
    palette="cubehelix"
)
plt.title("Cross-platform Deployment Challenges")
plt.tight_layout()
plt.show()

# =======================
# 4. Donut chart - Game Elements Alignment
# =======================
plt.figure(figsize=(6,6))
sizes = df[selected_questions[3]].value_counts()
plt.pie(
    sizes,
    labels=sizes.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette("pastel")
)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title("Game Elements & Motivation Alignment")
plt.tight_layout()
plt.show()

# =======================
# 5. Count plot - Social Features Impact
# =======================
plt.figure(figsize=(8,5))
sns.countplot(
    x=df[selected_questions[4]],
    order=df[selected_questions[4]].value_counts().index,
    palette="mako"
)
plt.title("Impact of Social Features on Engagement")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# =======================
# 6. Stacked bar chart - Fallback Mechanisms vs Scalability
# =======================
cross_tab = pd.crosstab(df[selected_questions[5]], df[selected_questions[6]])
cross_tab.plot(
    kind="bar",
    stacked=True,
    figsize=(8,5),
    colormap="viridis"
)
plt.title("Fallback Mechanisms vs Scalability")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =======================
# 7. Heatmap - Correlation Between Selected Questions
# =======================
encoded_df = df[selected_questions].apply(lambda col: col.astype('category').cat.codes)
plt.figure(figsize=(8,6))
sns.heatmap(encoded_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Survey Responses")
plt.tight_layout()
plt.show()
