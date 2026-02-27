import numpy as np

# 1) Load data using numpy.load
data = np.load("credit_score_fairness_data.npy")  # ändra sökväg om filen ligger annanstans

# 2) Columns: ["Protected attribute", "True credit worthiness", "Algorithm prediction"]
A = data[:, 0].astype(int)   # protected attribute (t.ex. 0/1)
Y = data[:, 1].astype(int)   # true label (0/1)
Yhat = data[:, 2].astype(int) # prediction (0/1)

def confusion_matrix_elements(y_true, y_pred):
    """Returnerar TP, TN, FP, FN."""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def safe_div(n, d):
    return n / d if d != 0 else np.nan

groups = np.unique(A)
metrics = {}

for g in groups:
    mask = (A == g)
    TP, TN, FP, FN = confusion_matrix_elements(Y[mask], Yhat[mask])

    # Equal Opportunity rate = TPR
    TPR = safe_div(TP, TP + FN)

    # False positive rate (för Equalized Odds)
    FPR = safe_div(FP, FP + TN)

    # Equalized Error Rates (vanligt i uppgifter = jämför FPR och FNR)
    FNR = safe_div(FN, TP + FN)

    metrics[g] = {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "TPR": TPR, "FPR": FPR, "FNR": FNR}

# Print confusion matrix + rates per group
for g in groups:
    m = metrics[g]
    print(f"\nGroup {g}")
    print(f"  TP={m['TP']}  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}")
    print(f"  TPR (Equal Opportunity) = {m['TPR']:.4f}")
    print(f"  FPR = {m['FPR']:.4f}")
    print(f"  FNR = {m['FNR']:.4f}")

# 4) Differences across groups (assuming 2 groups: 0 and 1)
if len(groups) == 2:
    g0, g1 = groups[0], groups[1]

    # Equal Opportunity difference = |TPR0 - TPR1|
    eq_opp_diff = abs(metrics[g0]["TPR"] - metrics[g1]["TPR"])

    # Equalized Odds differences = |TPR0-TPR1| and |FPR0-FPR1|
    tpr_diff = eq_opp_diff
    fpr_diff = abs(metrics[g0]["FPR"] - metrics[g1]["FPR"])

    # Equalized Error Rates differences = |FPR0-FPR1| and |FNR0-FNR1|
    fnr_diff = abs(metrics[g0]["FNR"] - metrics[g1]["FNR"])

    print("\n=== Fairness comparisons ===")
    print(f"Equal Opportunity difference (TPR diff): {eq_opp_diff:.4f}")
    print(f"Equalized Odds diffs: TPR diff={tpr_diff:.4f}, FPR diff={fpr_diff:.4f}")
    print(f"Equalized Error Rates diffs: FPR diff={fpr_diff:.4f}, FNR diff={fnr_diff:.4f}")

    # Simple fairness rule-of-thumb (du kan ändra tröskel)
    threshold = 0.05
    if tpr_diff < threshold and fpr_diff < threshold:
        print("\nConclusion: Roughly fair under Equalized Odds (small TPR & FPR differences).")
    else:
        print("\nConclusion: Not fair under Equalized Odds (large TPR and/or FPR difference).")
else:
    print("\nNote: More than 2 protected groups detected; compare pairwise if needed.")