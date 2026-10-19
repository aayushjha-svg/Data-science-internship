import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, json

sns.set_theme(style="whitegrid", palette="deep")
out = {}

# 1. ACQUISITION
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df_raw = pd.read_csv(URL)
df = df_raw.copy()
out["shape_raw"] = df.shape
out["dtypes_raw"] = df.dtypes.astype(str).to_dict()

# 2. INSPECTION
miss = df.isna().sum()
miss_pct = (miss / len(df) * 100).round(2)
out["missing"] = {c: [int(miss[c]), float(miss_pct[c])] for c in df.columns if miss[c] > 0}
out["duplicates_rows"] = int(df.duplicated().sum())
out["duplicates_id"] = int(df.PassengerId.duplicated().sum())
out["duplicates_name"] = int(df.Name.duplicated().sum())
out["describe_raw"] = df.describe().round(2).to_dict()

# FIG 1: missing values (bar + matrix)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios":[1,1.3]})
mp = miss_pct[miss_pct > 0].sort_values()
ax[0].barh(mp.index, mp.values, color="#c0504d")
for i, v in enumerate(mp.values): ax[0].text(v+0.8, i, f"{v:.1f}%", va="center")
ax[0].set_xlabel("% missing"); ax[0].set_title("Missing values by column"); ax[0].set_xlim(0, 100)
sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap=["#e8eef5", "#c0504d"], ax=ax[1])
ax[1].set_title("Missingness map (red = missing)")
plt.tight_layout(); plt.savefig("fig1_missing.png", dpi=150); plt.close()

# 3. CLEANING
# 3a duplicates
df = df.drop_duplicates()
# 3b strip whitespace in text columns
for c in ["Name","Sex","Ticket","Cabin","Embarked"]:
    df[c] = df[c].astype("string").str.strip()
# 3c title extraction (used for smarter age imputation)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
rare = ["Dr","Rev","Col","Major","Mlle","Mme","Don","Lady","Sir","Capt","the Countess","Jonkheer","Ms","Dona"]
df["Title"] = df["Title"].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})
vc = df["Title"].value_counts()
df["Title"] = df["Title"].where(df["Title"].map(vc) >= 10, "Rare")
out["titles"] = df["Title"].value_counts().to_dict()
# 3d Age: flag + median by Title x Pclass
df["AgeWasMissing"] = df["Age"].isna().astype(int)
med = df.groupby(["Title","Pclass"])["Age"].transform("median")
df["Age"] = df["Age"].fillna(med).fillna(df["Age"].median())
# 3e Embarked: mode
mode_emb = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(mode_emb)
out["embarked_mode"] = mode_emb
# 3f Cabin: too sparse -> HasCabin + Deck
df["HasCabin"] = df["Cabin"].notna().astype(int)
df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
df = df.drop(columns=["Cabin"])
# 3g types
for c in ["Survived","Pclass"]: df[c] = df[c].astype("int8")
for c in ["Sex","Embarked","Title","Deck"]: df[c] = df[c].astype("category")
df["Age"] = df["Age"].astype(float)
df["Embarked"] = df["Embarked"].cat.rename_categories({"S":"Southampton","C":"Cherbourg","Q":"Queenstown"})
# 3h outliers (IQR) on Fare - flag, keep
q1,q3 = df.Fare.quantile([.25,.75]); iqr=q3-q1
hi = q3+1.5*iqr
out["fare_iqr_upper"] = round(float(hi),2)
out["fare_outliers"] = int((df.Fare>hi).sum())
out["fare_zero"] = int((df.Fare==0).sum())
df["FareOutlier"] = (df.Fare>hi).astype(int)
# 3i feature engineering
df["FamilySize"] = df.SibSp + df.Parch + 1
df["IsAlone"] = (df.FamilySize==1).astype(int)
df["AgeGroup"] = pd.cut(df.Age, [0,12,18,35,60,100], labels=["Child","Teen","Young Adult","Adult","Senior"])
df["FareLog"] = np.log1p(df.Fare)
out["shape_clean"] = df.shape
out["missing_after"] = int(df.isna().sum().sum())
df.to_csv("titanic_clean.csv", index=False)

# 4. EDA
out["describe_clean"] = df[["Age","Fare","SibSp","Parch","FamilySize"]].describe().round(2).to_dict()
out["survival_overall"] = round(df.Survived.mean()*100,1)
out["surv_sex"] = (df.groupby("Sex",observed=True).Survived.mean()*100).round(1).to_dict()
out["surv_class"] = (df.groupby("Pclass").Survived.mean()*100).round(1).to_dict()
out["surv_embarked"] = (df.groupby("Embarked",observed=True).Survived.mean()*100).round(1).to_dict()
out["surv_agegroup"] = (df.groupby("AgeGroup",observed=True).Survived.mean()*100).round(1).to_dict()
out["surv_alone"] = (df.groupby("IsAlone").Survived.mean()*100).round(1).to_dict()
out["surv_famsize"] = (df.groupby("FamilySize").Survived.mean()*100).round(1).to_dict()
out["surv_title"] = (df.groupby("Title",observed=True).Survived.mean()*100).round(1).to_dict()
out["surv_hascabin"] = (df.groupby("HasCabin").Survived.mean()*100).round(1).to_dict()
out["surv_sex_class"] = (df.groupby(["Sex","Pclass"],observed=True).Survived.mean()*100).round(1).unstack().to_dict()
out["fare_by_class"] = df.groupby("Pclass").Fare.median().round(2).to_dict()
out["age_by_class"] = df.groupby("Pclass").Age.median().round(1).to_dict()
out["skew_fare"] = round(float(df.Fare.skew()),2); out["skew_farelog"]=round(float(df.FareLog.skew()),2)
corr = df[["Survived","Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone","HasCabin"]].corr()
out["corr_survived"] = corr["Survived"].round(3).to_dict()
out["corr_pclass_fare"] = round(float(corr.loc["Pclass","Fare"]),3)

# FIG 2: distributions
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
sns.histplot(df.Age, bins=30, kde=True, ax=ax[0], color="#4f81bd"); ax[0].set_title("Age distribution (after imputation)")
sns.histplot(df.Fare, bins=40, ax=ax[1], color="#9bbb59"); ax[1].set_title(f"Fare (raw, skew={out['skew_fare']})")
sns.histplot(df.FareLog, bins=40, ax=ax[2], color="#f79646"); ax[2].set_title(f"log(1+Fare) (skew={out['skew_farelog']})")
plt.tight_layout(); plt.savefig("fig2_distributions.png", dpi=150); plt.close()

# FIG 3: survival rates
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
sns.barplot(data=df, x="Sex", y="Survived", hue="Pclass", ax=ax[0], errorbar=None, palette="Blues_d")
ax[0].set_title("Survival rate by sex and class"); ax[0].set_ylabel("Survival rate"); ax[0].set_ylim(0,1)
sns.barplot(data=df, x="AgeGroup", y="Survived", ax=ax[1], errorbar=None, color="#4f81bd")
ax[1].set_title("Survival rate by age group"); ax[1].set_ylabel(""); ax[1].set_ylim(0,1); ax[1].tick_params(axis="x", rotation=20)
sns.barplot(data=df, x="FamilySize", y="Survived", ax=ax[2], errorbar=None, color="#c0504d")
ax[2].set_title("Survival rate by family size"); ax[2].set_ylabel(""); ax[2].set_ylim(0,1)
plt.tight_layout(); plt.savefig("fig3_survival.png", dpi=150); plt.close()

# FIG 4: box plot fare / age by class
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=df, x="Pclass", y="Fare", ax=ax[0], palette="Set2", hue="Pclass", legend=False); ax[0].set_yscale("log"); ax[0].set_title("Fare by class (log scale)")
sns.boxplot(data=df, x="Pclass", y="Age", ax=ax[1], palette="Set2", hue="Pclass", legend=False); ax[1].set_title("Age by class")
plt.tight_layout(); plt.savefig("fig4_boxplots.png", dpi=150); plt.close()

# FIG 5: correlation heatmap
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True)
ax.set_title("Correlation matrix (numeric features)")
plt.tight_layout(); plt.savefig("fig5_correlation.png", dpi=150); plt.close()

# FIG 6: scatter
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="Age", y="Fare", hue=df.Survived.map({0:"Died",1:"Survived"}), alpha=.65, ax=ax, palette={"Died":"#c0504d","Survived":"#4f81bd"})
ax.set_yscale("symlog", linthresh=10); ax.set_ylim(0, 600); ax.set_title("Age vs Fare coloured by survival")
plt.tight_layout(); plt.savefig("fig6_scatter.png", dpi=150); plt.close()

json.dump(out, open("results.json","w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
