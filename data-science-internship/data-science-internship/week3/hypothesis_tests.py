"""Week 3: Statistical analysis and hypothesis testing on the Gapminder dataset."""
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import Table2x2

ALPHA = 0.05
URL = "https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv"
df = pd.read_csv(URL)
df["logGDP"] = np.log10(df.gdpPercap)
d07 = df[df.year == 2007].copy()
wide = df.pivot(index="country", columns="year", values="lifeExp")
cont = df.drop_duplicates("country").set_index("country").continent
R = {}

PAL = {"Africa": "#D55E00", "Americas": "#009E73", "Asia": "#E69F00", "Europe": "#0072B2", "Oceania": "#CC79A7"}
INK, MUTED, GRID = "#222222", "#6B7280", "#E5E7EB"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "savefig.dpi": 170, "savefig.bbox": "tight",
    "savefig.facecolor": "white"})
def head(ax, a, b=None):
    ax.text(0, 1.13, a, transform=ax.transAxes, fontsize=12.5, fontweight="bold", color=INK)
    if b: ax.text(0, 1.06, b, transform=ax.transAxes, fontsize=9.3, color=MUTED)

# ============ TEST 1: One-way ANOVA, life expectancy by continent (2007) ============
grp = d07[d07.continent != "Oceania"]                      # Oceania has n = 2, excluded
order = ["Africa", "Americas", "Asia", "Europe"]
groups = [grp[grp.continent == c].lifeExp.values for c in order]
R["anova_n"] = {c: int(len(g)) for c, g in zip(order, groups)}
R["anova_mean"] = {c: round(float(g.mean()), 1) for c, g in zip(order, groups)}
R["anova_sd"] = {c: round(float(g.std(ddof=1)), 1) for c, g in zip(order, groups)}
R["shapiro_groups"] = {c: round(float(stats.shapiro(g).pvalue), 3) for c, g in zip(order, groups)}
lev = stats.levene(*groups, center="median"); R["levene"] = [round(float(lev.statistic), 2), float(lev.pvalue)]
f = stats.f_oneway(*groups); R["anova_F"] = [round(float(f.statistic), 1), float(f.pvalue)]
w = anova_oneway(groups, use_var="unequal", welch_correction=True); R["welch_F"] = [round(float(w.statistic), 1), float(w.pvalue)]
kw = stats.kruskal(*groups); R["kruskal"] = [round(float(kw.statistic), 1), float(kw.pvalue)]
ss_between = sum(len(g) * (g.mean() - grp.lifeExp.mean()) ** 2 for g in groups)
ss_total = ((grp.lifeExp - grp.lifeExp.mean()) ** 2).sum()
R["eta2"] = round(float(ss_between / ss_total), 3)
tk = pairwise_tukeyhsd(grp.lifeExp, grp.continent, alpha=ALPHA)
tkdf = pd.DataFrame(tk._results_table.data[1:], columns=tk._results_table.data[0])
R["tukey"] = [[r.group1, r.group2, round(float(r.meandiff), 1), round(float(r["p-adj"]), 4), round(float(r.lower), 1), round(float(r.upper), 1), bool(r.reject)] for _, r in tkdf.iterrows()]

# Fig 1
fig, ax = plt.subplots(figsize=(8.5, 5.2))
data = [grp[grp.continent == c].lifeExp for c in order]
bp = ax.boxplot(data, positions=range(4), widths=.5, patch_artist=True, showfliers=False,
                medianprops=dict(color="white", lw=2), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
for patch, c in zip(bp["boxes"], order): patch.set(facecolor=PAL[c], alpha=.85, edgecolor="white")
rng = np.random.default_rng(1)
for i, c in enumerate(order):
    y = data[i].values; ax.scatter(i + rng.uniform(-.17, .17, len(y)), y, s=14, color="#222", alpha=.45, zorder=3)
    m = y.mean(); se = y.std(ddof=1) / np.sqrt(len(y)); ci = stats.t.ppf(.975, len(y) - 1) * se
    ax.errorbar(i + .36, m, yerr=ci, fmt="D", color=INK, ms=6, capsize=4, zorder=5)
    ax.text(i, 86, f"n={len(y)}", ha="center", fontsize=9, color=MUTED)
ax.errorbar([], [], yerr=[], fmt="D", color=INK, label="Mean with 95% CI")
ax.legend(loc="lower right", frameon=False)
ax.set_xticks(range(4)); ax.set_xticklabels(order); ax.set_ylim(35, 88); ax.set_ylabel("Life expectancy 2007 (years)")
head(ax, "Test 1: Life expectancy differs by continent", f"One-way ANOVA: F = {R['anova_F'][0]}, eta-squared = {R['eta2']}")
plt.savefig("fig1_anova.png"); plt.close()

# ============ TEST 2: Paired t-test, 1952 vs 2007 ============
a, b = wide[1952].values, wide[2007].values
diff = b - a
R["paired_mean_diff"] = round(float(diff.mean()), 2); R["paired_sd_diff"] = round(float(diff.std(ddof=1)), 2)
R["shapiro_diff"] = [round(float(stats.shapiro(diff).statistic), 3), float(stats.shapiro(diff).pvalue)]
R["skew_diff"] = round(float(stats.skew(diff)), 2)
tt = stats.ttest_rel(b, a); ci = tt.confidence_interval()
R["paired_t"] = [round(float(tt.statistic), 2), float(tt.pvalue), int(tt.df)]
R["paired_ci"] = [round(float(ci.low), 2), round(float(ci.high), 2)]
R["paired_dz"] = round(float(diff.mean() / diff.std(ddof=1)), 2)
wx = stats.wilcoxon(b, a); R["wilcoxon"] = [float(wx.statistic), float(wx.pvalue)]
R["n_countries"] = int(len(diff)); R["n_improved"] = int((diff > 0).sum())
# Fig 2
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.hist(diff, bins=24, color="#3E7CB1", edgecolor="white", alpha=.9)
ax.axvline(0, color="#B91C1C", lw=2, ls="--"); ax.text(.3, ax.get_ylim()[1] * .94, "No change", color="#B91C1C", fontsize=9)
ax.axvspan(ci.low, ci.high, color="#F59E0B", alpha=.35)
ax.axvline(diff.mean(), color=INK, lw=2)
ax.text(diff.mean() + .5, ax.get_ylim()[1] * .86, f"Mean gain {diff.mean():.1f} yrs\n95% CI [{ci.low:.1f}, {ci.high:.1f}]", fontsize=9.5, color=INK)
ax.set_xlabel("Change in life expectancy, 1952 to 2007 (years)"); ax.set_ylabel("Number of countries"); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
head(ax, "Test 2: Life expectancy rose in almost every country", f"Paired t-test on {len(diff)} countries: t = {tt.statistic:.1f}, Cohen's dz = {R['paired_dz']}")
plt.savefig("fig2_paired.png"); plt.close()

# ============ TEST 3: Welch t-test, gain in Africa vs other continents ============
gain = (wide[2007] - wide[1952]); isAf = (cont.reindex(gain.index) == "Africa")
g_af, g_ot = gain[isAf].values, gain[~isAf].values
R["gain_af"] = [int(len(g_af)), round(float(g_af.mean()), 1), round(float(g_af.std(ddof=1)), 1)]
R["gain_ot"] = [int(len(g_ot)), round(float(g_ot.mean()), 1), round(float(g_ot.std(ddof=1)), 1)]
lv = stats.levene(g_af, g_ot, center="median"); R["levene_gain"] = [round(float(lv.statistic), 2), round(float(lv.pvalue), 3)]
wt = stats.ttest_ind(g_ot, g_af, equal_var=False); wci = wt.confidence_interval()
R["welch"] = [round(float(wt.statistic), 2), float(wt.pvalue), round(float(wt.df), 1)]
R["welch_diff"] = round(float(g_ot.mean() - g_af.mean()), 2); R["welch_ci"] = [round(float(wci.low), 2), round(float(wci.high), 2)]
sp = np.sqrt(((len(g_af) - 1) * g_af.var(ddof=1) + (len(g_ot) - 1) * g_ot.var(ddof=1)) / (len(g_af) + len(g_ot) - 2))
R["welch_d"] = round(float((g_ot.mean() - g_af.mean()) / sp), 2)
mw = stats.mannwhitneyu(g_ot, g_af, alternative="two-sided"); R["mannwhitney"] = [float(mw.statistic), float(mw.pvalue)]
# Fig 3
fig, ax = plt.subplots(figsize=(8.5, 4.8))
bins = np.linspace(-6, 42, 25)
ax.hist(g_ot, bins=bins, color="#0072B2", alpha=.75, label=f"Other continents (n={len(g_ot)})", edgecolor="white")
ax.hist(g_af, bins=bins, color="#D55E00", alpha=.75, label=f"Africa (n={len(g_af)})", edgecolor="white")
for g, c in [(g_ot, "#0072B2"), (g_af, "#D55E00")]:
    ax.axvline(g.mean(), color=c, lw=2.2, ls="--")
ax.legend(frameon=False); ax.set_xlabel("Gain in life expectancy 1952 to 2007 (years)"); ax.set_ylabel("Number of countries"); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
head(ax, "Test 3: African countries gained less on average", f"Welch t-test: t = {wt.statistic:.1f}, mean difference = {R['welch_diff']} years (dashed lines = group means)")
plt.savefig("fig3_welch.png"); plt.close()

# ============ TEST 4: Correlation + regression, income vs life expectancy (2007) ============
pr = stats.pearsonr(d07.logGDP, d07.lifeExp); prci = pr.confidence_interval()
sr = stats.spearmanr(d07.logGDP, d07.lifeExp)
R["pearson"] = [round(float(pr.statistic), 3), float(pr.pvalue), round(float(prci.low), 3), round(float(prci.high), 3)]
R["spearman"] = [round(float(sr.statistic), 3), float(sr.pvalue)]
X = sm.add_constant(d07.logGDP); ols = sm.OLS(d07.lifeExp, X).fit()
R["ols_slope"] = round(float(ols.params["logGDP"]), 2); R["ols_slope_ci"] = [round(float(x), 2) for x in ols.conf_int().loc["logGDP"]]
R["ols_r2"] = round(float(ols.rsquared), 3); R["ols_n"] = int(ols.nobs)
R["per_doubling"] = round(float(ols.params["logGDP"] * np.log10(2)), 2)
R["per_doubling_ci"] = [round(float(x * np.log10(2)), 2) for x in ols.conf_int().loc["logGDP"]]
from statsmodels.stats.diagnostic import het_breuschpagan
R["bp_p"] = round(float(het_breuschpagan(ols.resid, X)[1]), 3)
R["shapiro_resid_p"] = round(float(stats.shapiro(ols.resid).pvalue), 3)
# Fig 4
fig, ax = plt.subplots(figsize=(8.5, 5.2))
xs = np.linspace(d07.logGDP.min(), d07.logGDP.max(), 100)
pred = ols.get_prediction(sm.add_constant(pd.Series(xs, name="logGDP"))).summary_frame(alpha=.05)
ax.fill_between(10 ** xs, pred.mean_ci_lower, pred.mean_ci_upper, color="#F59E0B", alpha=.45, label="95% CI of the mean")
ax.plot(10 ** xs, pred["mean"], color=INK, lw=2)
ax.scatter(d07.gdpPercap, d07.lifeExp, c=d07.continent.map(PAL), s=34, alpha=.85, edgecolor="white", linewidth=.5, zorder=3)
for k, v in PAL.items(): ax.scatter([], [], c=v, s=34, label=k)
ax.set_xscale("log"); ax.set_xticks([500, 1000, 2000, 5000, 10000, 20000, 50000]); ax.set_xticklabels(["$500", "$1k", "$2k", "$5k", "$10k", "$20k", "$50k"])
ax.legend(frameon=False, loc="lower right", ncol=2); ax.set_xlabel("Income per person 2007 (US$, log scale)"); ax.set_ylabel("Life expectancy (years)")
head(ax, "Test 4: Higher income goes with longer life", f"Pearson r = {pr.statistic:.2f}, 95% CI [{prci.low:.2f}, {prci.high:.2f}]; R-squared = {R['ols_r2']}")
plt.savefig("fig4_regression.png"); plt.close()

# ============ TEST 5: Chi-square, income group vs high life expectancy (2007) ============
med = d07.gdpPercap.median(); R["median_income"] = round(float(med))
d07["incomeGroup"] = np.where(d07.gdpPercap > med, "Above-median income", "Below-median income")
d07["leGroup"] = np.where(d07.lifeExp >= 70, "Life exp. 70+", "Life exp. under 70")
ct = pd.crosstab(d07.incomeGroup, d07.leGroup).loc[["Above-median income", "Below-median income"], ["Life exp. 70+", "Life exp. under 70"]]
R["ct"] = ct.values.tolist()
chi = stats.chi2_contingency(ct, correction=False); chic = stats.chi2_contingency(ct, correction=True)
R["chi2"] = [round(float(chi.statistic), 2), float(chi.pvalue), int(chi.dof)]
R["chi2_yates"] = [round(float(chic.statistic), 2), float(chic.pvalue)]
R["expected"] = np.round(chi.expected_freq, 1).tolist()
R["cramers_v"] = round(float(np.sqrt(chi.statistic / ct.values.sum())), 2)
t22 = Table2x2(ct.values); R["odds_ratio"] = round(float(t22.oddsratio), 1)
R["or_ci"] = [round(float(x), 1) for x in t22.oddsratio_confint()]
fe = stats.fisher_exact(ct.values); R["fisher_p"] = float(fe.pvalue)
R["share_high"] = [round(float(ct.iloc[0, 0] / ct.iloc[0].sum() * 100), 1), round(float(ct.iloc[1, 0] / ct.iloc[1].sum() * 100), 1)]
# Fig 5
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1]})
ax = axs[0]
prop = ct.div(ct.sum(axis=1), axis=0) * 100
ax.bar([0, 1], prop["Life exp. 70+"], color="#0B6E4F", label="Life expectancy 70+", width=.55)
ax.bar([0, 1], prop["Life exp. under 70"], bottom=prop["Life exp. 70+"], color="#D9D9D9", label="Under 70", width=.55)
for i in range(2):
    ax.text(i, prop["Life exp. 70+"].iloc[i] / 2, f"{prop['Life exp. 70+'].iloc[i]:.0f}%", ha="center", color="white", fontweight="bold")
    ax.text(i, prop["Life exp. 70+"].iloc[i] + prop["Life exp. under 70"].iloc[i] / 2, f"{prop['Life exp. under 70'].iloc[i]:.0f}%", ha="center", color=INK)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Above-median\nincome", "Below-median\nincome"]); ax.set_ylabel("% of countries"); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2)
head(ax, "Test 5: Income group and longevity are linked", f"Chi-square = {chi.statistic:.1f}, Cramer's V = {R['cramers_v']}")
ax = axs[1]
obs = ct.values.flatten(); exp = chi.expected_freq.flatten()
labs = ["High inc.\n70+", "High inc.\n<70", "Low inc.\n70+", "Low inc.\n<70"]
xx = np.arange(4); ax.bar(xx - .18, obs, .36, color="#0072B2", label="Observed"); ax.bar(xx + .18, exp, .36, color="#BBBBBB", label="Expected if unrelated")
ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=8.5); ax.set_ylabel("Number of countries"); ax.legend(frameon=False); ax.grid(axis="x", visible=False)
head(ax, "Observed vs expected counts", "Large gaps = the variables are not independent")
plt.tight_layout(); plt.savefig("fig5_chisq.png"); plt.close()

# ============ Assumption checks figure ============
fig, axs = plt.subplots(1, 3, figsize=(12, 3.9))
for ax, (y, ttl) in zip(axs, [(diff, "Paired differences (Test 2)"), (sm.OLS(grp.lifeExp, sm.add_constant(pd.get_dummies(grp.continent, drop_first=True).astype(float))).fit().resid, "ANOVA residuals (Test 1)"), (ols.resid, "Regression residuals (Test 4)")]):
    (osm, osr), (sl, ic, r) = stats.probplot(y, dist="norm")
    ax.scatter(osm, osr, s=14, color="#3E7CB1", alpha=.7); ax.plot(osm, sl * osm + ic, color="#B91C1C", lw=1.6)
    ax.set_title(ttl, fontsize=10.5, fontweight="bold"); ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
plt.tight_layout(); plt.savefig("fig6_assumptions.png"); plt.close()

# ============ Multiple testing correction (Holm) over the 5 primary tests ============
raw = {"ANOVA (continent)": R["welch_F"][1], "Paired t (1952 vs 2007)": R["paired_t"][1], "Welch t (Africa vs others)": R["welch"][1],
       "Pearson (income vs life exp.)": R["pearson"][1], "Chi-square (income group)": R["chi2"][1]}
rej, adj, _, _ = multipletests(list(raw.values()), alpha=ALPHA, method="holm")
R["holm"] = {k: [float(v), float(a), bool(r)] for (k, v), a, r in zip(raw.items(), adj, rej)}

json.dump(R, open("results3.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
