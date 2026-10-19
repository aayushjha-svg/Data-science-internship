"""Week 4: Classification model - will a country have life expectancy of 70+ years? (Gapminder)"""
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

SEED = 42
URL = "https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv"
df = pd.read_csv(URL)

# ---------- 1. DATA PREPARATION ----------
assert df.isna().sum().sum() == 0 and df.duplicated().sum() == 0
df["logGDP"] = np.log10(df.gdpPercap)
df["logPop"] = np.log10(df["pop"])
df["High"] = (df.lifeExp >= 70).astype(int)                 # target: life expectancy of 70+ years
FEATS_NUM, FEATS_CAT = ["logGDP", "logPop", "year"], ["continent"]
X, y, groups = df[FEATS_NUM + FEATS_CAT], df.High, df.country
R = {"n_rows": int(len(df)), "n_countries": int(df.country.nunique()), "pos_rate": round(float(y.mean() * 100), 1)}
R["pos_rate_by_year"] = {int(k): round(float(v * 100)) for k, v in df.groupby("year").High.mean().items()}

# ---------- 2. SPLIT: by COUNTRY, so a country never appears in both train and test ----------
gss = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED)
tr, te = next(gss.split(X, y, groups))
Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
R["n_train"], R["n_test"] = int(len(tr)), int(len(te))
R["countries_train"], R["countries_test"] = int(groups.iloc[tr].nunique()), int(groups.iloc[te].nunique())
R["pos_train"], R["pos_test"] = round(float(ytr.mean() * 100), 1), round(float(yte.mean() * 100), 1)

pre = ColumnTransformer([("num", StandardScaler(), FEATS_NUM), ("cat", OneHotEncoder(drop="first"), FEATS_CAT)])
models = {
    "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": LogisticRegression(max_iter=1000),
    "Decision tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=SEED),
    "Random forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=SEED),
}
pipes = {k: Pipeline([("pre", pre), ("clf", m)]) for k, m in models.items()}

# ---------- 3. CROSS-VALIDATION on the training set (grouped by country) ----------
gkf = GroupKFold(n_splits=5)
R["cv"] = {}
for k, pp in pipes.items():
    cv = cross_validate(pp, Xtr, ytr, groups=groups.iloc[tr], cv=gkf, scoring=["accuracy", "roc_auc"])
    R["cv"][k] = [round(float(cv["test_accuracy"].mean()), 3), round(float(cv["test_accuracy"].std()), 3),
                  round(float(cv["test_roc_auc"].mean()), 3), round(float(cv["test_roc_auc"].std()), 3)]

# ---------- 4. FIT + TEST EVALUATION ----------
res, probs, preds = {}, {}, {}
for k, pp in pipes.items():
    pp.fit(Xtr, ytr)
    pr = pp.predict_proba(Xte)[:, 1]; pd_ = pp.predict(Xte)
    probs[k], preds[k] = pr, pd_
    res[k] = dict(acc=accuracy_score(yte, pd_), prec=precision_score(yte, pd_, zero_division=0), rec=recall_score(yte, pd_),
                  f1=f1_score(yte, pd_), auc=roc_auc_score(yte, pr), train_acc=accuracy_score(ytr, pp.predict(Xtr)))
R["test"] = {k: {m: round(float(v), 3) for m, v in d.items()} for k, d in res.items()}
R["cm"] = {k: confusion_matrix(yte, preds[k]).tolist() for k in preds}

# ---------- 5. FIGURES ----------
INK, MUTED, GRID = "#222222", "#6B7280", "#E5E7EB"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "savefig.dpi": 170, "savefig.bbox": "tight", "savefig.facecolor": "white"})
COL = {"Baseline (majority class)": "#9CA3AF", "Logistic regression": "#0072B2", "Decision tree (depth 3)": "#E69F00", "Random forest": "#009E73"}
def head(ax, a, b=None):
    ax.text(0, 1.13, a, transform=ax.transAxes, fontsize=12.5, fontweight="bold", color=INK)
    if b: ax.text(0, 1.06, b, transform=ax.transAxes, fontsize=9.3, color=MUTED)

# Fig 1: class balance
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
yrs = sorted(R["pos_rate_by_year"]); axs[0].bar([str(v) for v in yrs], [R["pos_rate_by_year"][v] for v in yrs], color="#0B6E4F")
axs[0].set_ylabel("% of countries with life expectancy 70+"); axs[0].tick_params(axis="x", rotation=45); axs[0].grid(axis="x", visible=False)
head(axs[0], "The target changes over time", "Share of countries in the positive class, by year")
cnt = df.groupby("continent").High.mean() * 100
axs[1].barh(cnt.index, cnt.values, color=[{"Africa": "#D55E00", "Americas": "#009E73", "Asia": "#E69F00", "Europe": "#0072B2", "Oceania": "#CC79A7"}[c] for c in cnt.index])
axs[1].set_xlabel("% of country-years with life expectancy 70+"); axs[1].grid(axis="y", visible=False)
head(axs[1], "...and differs by continent", "Share of country-years in the positive class")
plt.tight_layout(); plt.savefig("fig1_target.png"); plt.close()

# Fig 2: ROC curves
fig, ax = plt.subplots(figsize=(6.6, 5.6))
for k in probs:
    fpr, tpr, _ = roc_curve(yte, probs[k]); ax.plot(fpr, tpr, lw=2.4 if k != "Baseline (majority class)" else 1.5, color=COL[k],
        ls="--" if k.startswith("Base") else "-", label=f"{k} (AUC = {res[k]['auc']:.3f})")
ax.plot([0, 1], [0, 1], color="#D1D5DB", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(frameon=False, loc="lower right", fontsize=8.5)
head(ax, "ROC curves on unseen countries", f"Test set: {R['n_test']} country-years from {R['countries_test']} held-out countries")
plt.savefig("fig2_roc.png"); plt.close()

# Fig 3: confusion matrices
fig, axs = plt.subplots(1, 3, figsize=(12, 3.9))
for ax, k in zip(axs, ["Logistic regression", "Decision tree (depth 3)", "Random forest"]):
    ConfusionMatrixDisplay(confusion_matrix(yte, preds[k]), display_labels=["Under 70", "70+"]).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{k}\naccuracy {res[k]['acc']:.1%}", fontsize=10.5, fontweight="bold"); ax.grid(False); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("fig3_confusion.png"); plt.close()

# Fig 4: metric comparison
fig, ax = plt.subplots(figsize=(9, 4.6))
mets = ["acc", "prec", "rec", "f1", "auc"]; labs = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
wid = .2
for i, k in enumerate(res):
    ax.bar(np.arange(5) + (i - 1.5) * wid, [res[k][m] for m in mets], wid, label=k, color=COL[k])
ax.set_xticks(range(5)); ax.set_xticklabels(labs); ax.set_ylim(0, 1.08); ax.legend(frameon=False, ncol=2, fontsize=8.5, loc="lower center", bbox_to_anchor=(.5, -.32)); ax.grid(axis="x", visible=False)
head(ax, "Model comparison on the test set", "Higher is better; the baseline always predicts the majority class")
plt.savefig("fig4_metrics.png"); plt.close()

# Fig 5: overfitting check - tree depth vs train/CV accuracy
depths = range(1, 13); trn, cvs = [], []
for dpt in depths:
    pp = Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(max_depth=dpt, random_state=SEED))])
    trn.append(pp.fit(Xtr, ytr).score(Xtr, ytr))
    cvs.append(cross_val_score(pp, Xtr, ytr, groups=groups.iloc[tr], cv=gkf, scoring="accuracy").mean())
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(depths, trn, marker="o", color="#B91C1C", label="Training accuracy"); ax.plot(depths, cvs, marker="o", color="#0072B2", label="Cross-validated accuracy")
bd = int(list(depths)[int(np.argmax(cvs))]); ax.axvline(bd, color=MUTED, ls=":"); ax.text(bd + .15, .905, f"best CV depth = {bd}", fontsize=9, color=MUTED)
ax.fill_between(depths, cvs, trn, color="#FECACA", alpha=.35); ax.text(8.2, (trn[8] + cvs[8]) / 2 - .002, "overfitting gap", fontsize=9, color="#B91C1C")
ax.set_xlabel("Maximum tree depth"); ax.set_ylabel("Accuracy"); ax.legend(frameon=False, loc="center left")
head(ax, "Overfitting check: deeper trees memorize the training data", "Training accuracy keeps rising; accuracy on unseen countries stops improving")
plt.savefig("fig5_overfit.png"); plt.close()
R["overfit_best_depth"] = bd; R["overfit_train_cv"] = [[int(d_), round(float(t_), 3), round(float(c_), 3)] for d_, t_, c_ in zip(depths, trn, cvs)]

# Fig 6: coefficients (logistic) + forest importance
lr = pipes["Logistic regression"]; names = list(lr["pre"].get_feature_names_out())
clean = [n.replace("num__", "").replace("cat__continent_", "continent: ") for n in names]
coef = lr["clf"].coef_[0]; order = np.argsort(np.abs(coef))
rf = pipes["Random forest"]; imp = rf["clf"].feature_importances_; o2 = np.argsort(imp)
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
axs[0].barh(np.array(clean)[order], coef[order], color=["#0B6E4F" if c > 0 else "#B91C1C" for c in coef[order]])
axs[0].axvline(0, color=MUTED, lw=.8); axs[0].set_xlabel("Coefficient (standardised features)"); axs[0].grid(axis="y", visible=False)
head(axs[0], "Logistic regression coefficients", "Positive = raises the odds of life expectancy 70+")
axs[1].barh(np.array(clean)[o2], imp[o2], color="#009E73"); axs[1].set_xlabel("Importance"); axs[1].grid(axis="y", visible=False)
head(axs[1], "Random forest feature importance", "Share of the model's splitting decisions")
plt.tight_layout(); plt.savefig("fig6_features.png"); plt.close()
R["coef"] = {c: round(float(v), 2) for c, v in zip(clean, coef)}; R["imp"] = {c: round(float(v), 3) for c, v in zip(clean, imp)}

# ---------- 6. ERROR ANALYSIS (best model on test set) ----------
best = max([k for k in res if not k.startswith("Base")], key=lambda k: res[k]["auc"]); R["best"] = best
err = Xte.copy(); err["country"] = df.country.iloc[te].values; err["actual"] = yte.values; err["pred"] = preds[best]; err["p"] = probs[best]
err["lifeExp"] = df.lifeExp.iloc[te].values
wrong = err[err.actual != err.pred]
R["n_wrong"] = int(len(wrong)); R["n_fp"] = int(((wrong.pred == 1)).sum()); R["n_fn"] = int(((wrong.pred == 0)).sum())
R["wrong_by_country"] = wrong.country.value_counts().head(6).to_dict()
R["wrong_by_continent"] = wrong.continent.value_counts().to_dict()
R["wrong_detail"] = {c: {"FP": int(((g.pred == 1)).sum()), "FN": int(((g.pred == 0)).sum()), "years": sorted(g.year.astype(int).tolist())} for c, g in wrong[wrong.country.isin(wrong.country.value_counts().head(6).index)].groupby("country")}
near = (err.lifeExp.between(67, 73)); R["wrong_near_threshold"] = int((near & (err.actual != err.pred)).sum()); R["near_threshold_total"] = int(near.sum())
R["share_wrong_near"] = round(float(((wrong.lifeExp.between(66, 74)).mean()) * 100), 1)
R["wrong_by_year_bucket"] = wrong.assign(b=np.where(wrong.year < 1980, "before 1980", "1980+")).b.value_counts().to_dict()
R["test_by_year_bucket"] = err.assign(b=np.where(err.year < 1980, "before 1980", "1980+")).b.value_counts().to_dict()
# Fig 7: error analysis - predicted probability vs actual life expectancy
fig, ax = plt.subplots(figsize=(8.8, 5))
ok = err[err.actual == err.pred]; ax.scatter(ok.lifeExp, ok.p, s=16, color="#9CA3AF", alpha=.55, label="Correct")
ax.scatter(wrong.lifeExp, wrong.p, s=46, color="#B91C1C", edgecolor="white", label="Misclassified", zorder=5)
ax.axvline(70, color=INK, ls="--", lw=1); ax.axhline(.5, color=MUTED, ls=":", lw=1)
ax.set_xlabel("Actual life expectancy (years)"); ax.set_ylabel(f"Predicted probability of 70+ ({best})"); ax.legend(frameon=False, loc="center left")
head(ax, "Where the model goes wrong", "Most mistakes fall within 4 years of the 70-year threshold, but a few are far off")
plt.savefig("fig7_errors.png"); plt.close()

# ---------- 7. ROBUSTNESS: forward-in-time split (train up to 1987, test 1992-2007) ----------
trm, tem = df.year <= 1987, df.year >= 1992
pp = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))]).fit(X[trm], y[trm])
pt = pp.predict_proba(X[tem])[:, 1]
R["temporal"] = {"acc": round(float(accuracy_score(y[tem], pt > .5)), 3), "auc": round(float(roc_auc_score(y[tem], pt)), 3),
                 "recall": round(float(recall_score(y[tem], pt > .5)), 3), "base": round(float(max(y[tem].mean(), 1 - y[tem].mean())), 3)}
pp_rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=SEED))]).fit(X[trm], y[trm])
R["temporal_rf"] = {"acc": round(float(accuracy_score(y[tem], pp_rf.predict(X[tem]))), 3), "auc": round(float(roc_auc_score(y[tem], pp_rf.predict_proba(X[tem])[:, 1])), 3)}
json.dump(R, open("results4.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
