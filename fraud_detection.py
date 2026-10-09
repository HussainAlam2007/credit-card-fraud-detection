import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)


print("STEP 1: Loading real data file...")

df = pd.read_csv("creditcard.csv")

print("Here is a peek at the data (first 5 rows):")
print(df.head())

print(f"\nTotal transactions: {len(df)}")
print(f"Fraud cases: {df['Class'].sum()}")
print(f"Normal cases: {len(df) - df['Class'].sum()}")
print(f"Fraud percentage: {df['Class'].mean() * 100:.3f}%  <-- fraud is VERY rare!")


print("STEP 2: Making a chart to LOOK at the data...")

plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df, hue="Class", palette=["green", "red"], legend=False)
plt.title("Normal (0) vs Fraud (1) Transaction Counts")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.yscale("log")  # log scale, otherwise the fraud bar is invisible next to normal
plt.tight_layout()
plt.savefig("chart_1_class_counts.png", dpi=150)
plt.show()
plt.close()

print("Saved: chart_1_class_counts.png")
print("(We used a 'log scale' on this chart, otherwise the tiny fraud bar")
print(" would be invisible next to the huge normal bar!)")



print("STEP 3: Preparing the data...")

scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

feature_columns = [col for col in df.columns if col.startswith("V")] + ["Amount_scaled"]
X = df[feature_columns]   # the clues
y = df["Class"]            # the answer: fraud or not

print(f"Using {len(feature_columns)} clues per transaction (the V columns + Amount).")


print("STEP 4: Splitting into training (practice) and testing (quiz) sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    # stratify=y keeps the same fraud % in both the practice and quiz piles
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")



print("STEP 5: Teaching the computer using Logistic Regression...")

# class_weight="balanced" tells the model: "fraud cases are rare, so pay
# EXTRA attention to them, don't just ignore them because they're few."
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

print("Done! The computer has learned the pattern.")


print("STEP 6: Testing the computer on data it has NEVER seen...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("(Remember: accuracy alone can be misleading when fraud is this rare!)")

print("\nFull report (precision & recall for each class):")
print(classification_report(y_test, predictions, target_names=["Normal", "Fraud"]))
print("Recall for Fraud = out of all REAL fraud cases, what % did we catch?")
print("Precision for Fraud = when we flagged something as fraud, what % were actually fraud?")



print("STEP 7: Making a chart to show how well it did...")

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Normal", "Predicted Fraud"],
            yticklabels=["Actual Normal", "Actual Fraud"])
plt.title("How Well Did We Do?")
plt.tight_layout()
plt.savefig("chart_2_results.png", dpi=150)
plt.show()
plt.close()

print("Saved: chart_2_results.png")


print("ALL DONE! Check the two PNG chart files that were saved.")
