import pandas as pd, numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------- 1. LOAD + CHECK ----------
URL = "https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv"
df = pd.read_csv(URL)
assert df.isna().sum().sum() == 0 and df.duplicated().sum() == 0
df["logGDP"] = np.log10(df.gdpPercap)

# ---------- 2. STYLE ----------
PAL = {"Africa": "#D55E00", "Americas": "#009E73", "Asia": "#E69F00", "Europe": "#0072B2", "Oceania": "#CC79A7"}
INK, MUTED, GRID = "#222222", "#6B7280", "#E5E7EB"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": .7, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight", "savefig.facecolor": "white"})
def title(ax, head, sub=None):
    ax.text(0, 1.14, head, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, va="bottom")
    if sub: ax.text(0, 1.07, sub, transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")
xt = [500, 1000, 2000, 5000, 10000, 20000, 50000]
R = {}

p = df.pivot(index="country", columns="year", values="lifeExp")
gp = df.pivot(index="country", columns="year", values="gdpPercap")
pop = df.pivot(index="country", columns="year", values="pop")
cont = df.drop_duplicates("country").set_index("country").continent

# ---------- FIG 1: bubble 1952 vs 2007 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
lab = {1952: ["China", "India", "United States", "Afghanistan", "Japan", "Nigeria"],
       2007: ["China", "India", "United States", "Nigeria", "Japan", "Zimbabwe", "Afghanistan", "Norway"]}
off = {"China": (-30, -32), "India": (12, -20), "United States": (-45, 12), "Afghanistan": (8, -4), "Japan": (-8, 14),
       "Nigeria": (8, -14), "Zimbabwe": (6, -14), "Norway": (-10, 12)}
for ax, yr in zip(axs, [1952, 2007]):
    s = df[df.year == yr].sort_values("pop", ascending=False)
    ax.scatter(s.gdpPercap, s.lifeExp, s=s["pop"] / 1.1e6 + 14, c=s.continent.map(PAL), alpha=.72, edgecolor="white", linewidth=.7)
    ax.set_xscale("log"); ax.set_xticks(xt); ax.set_xticklabels([f"${x:,}" for x in xt], fontsize=8)
    ax.set_xlim(200, 60000); ax.set_ylim(20, 88)
    ax.text(.03, .94, str(yr), transform=ax.transAxes, fontsize=34, fontweight="bold", color="#D1D5DB", va="top")
    for c in lab[yr]:
        r = s[s.country == c].iloc[0]
        ax.annotate(c, (r.gdpPercap, r.lifeExp), xytext=off[c], textcoords="offset points", fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=.6))
    ax.set_xlabel("Income per person (US$, log scale)")
axs[0].set_ylabel("Life expectancy (years)")
for k, v in PAL.items(): axs[1].scatter([], [], c=v, s=60, label=k)
axs[1].legend(loc="lower right", frameon=False, title="Bubble size = population", title_fontsize=8.5)
title(axs[0], "The world moved up and to the right", "Each bubble is a country. Compare the cloud in 1952 (left) with 2007 (right).")
plt.tight_layout(); plt.savefig("fig1_bubble.png"); plt.close()
R["n_countries"] = int(df.country.nunique())
R["pop_weighted_le"] = {int(y): round(float(np.average(g.lifeExp, weights=g["pop"])), 1) for y, g in df.groupby("year")}
R["median_le"] = {int(y): round(float(p[y].median()), 1) for y in [1952, 2007]}
R["below50_countries"] = {int(y): int((p[y] < 50).sum()) for y in [1952, 2007]}
R["below50_popshare"] = {int(y): round(float(pop[y][p[y] < 50].sum() / pop[y].sum() * 100), 1) for y in [1952, 2007]}
R["gdp_median"] = {int(y): round(float(gp[y].median())) for y in [1952, 2007]}

# ---------- FIG 2: ridgeline ----------
years = [1952, 1967, 1982, 1997, 2007]
x = np.linspace(20, 90, 400)
fig, ax = plt.subplots(figsize=(9, 5.8))
ramp = ["#9CA3AF", "#6B8FB5", "#3E7CB1", "#1D5F9E", "#0B3C75"]
for i, yr in enumerate(years):
    y = gaussian_kde(p[yr], bw_method=.25)(x); y = y / y.max() * 1.55
    base = (len(years) - 1 - i) * 1.0
    ax.fill_between(x, base, base + y, color=ramp[i], alpha=.88, zorder=10 - i, linewidth=1.2, edgecolor="white")
    m = p[yr].median()
    ax.plot([m, m], [base, base + float(np.interp(m, x, y))], color="white", lw=1.6, zorder=11 - i)
    ax.text(21, base + .12, str(yr), fontsize=12, fontweight="bold", color=INK, zorder=20)
    ax.text(88.5, base + .12, f"median {m:.0f}", fontsize=8.5, color=("white" if i >= 2 else INK), ha="right", zorder=20)
ax.set_yticks([]); ax.set_xlim(20, 90); ax.grid(axis="y", visible=False)
ax.set_xlabel("Life expectancy at birth (years)"); ax.spines["left"].set_visible(False)
title(ax, "A rising tide, but a long tail was left behind", "Spread of life expectancy across 142 countries (white line = median country)")
n = len(p)
ax.text(91.5, 4.95, f"1952: {(p[1952]<50).sum()} of {n} countries\n({(p[1952]<50).sum()/n*100:.0f}%) were below 50 years", fontsize=9, color=INK, va="center", clip_on=False)
ax.text(91.5, 0.95, f"2007: only {(p[2007]<50).sum()} of {n} countries\n({(p[2007]<50).sum()/n*100:.0f}%) are below 50,\nbut that tail is still there", fontsize=9, color=INK, va="center", clip_on=False)
plt.savefig("fig2_ridgeline.png"); plt.close()
R["sd_le"] = {int(y): round(float(p[y].std()), 1) for y in [1952, 2007]}
R["min_max_le"] = {int(y): [round(float(p[y].min()), 1), round(float(p[y].max()), 1)] for y in [1952, 2007]}
R["gap_best_worst"] = {int(y): round(float(p[y].max() - p[y].min()), 1) for y in [1952, 2007]}

# ---------- FIG 3: dumbbell ----------
gain = (p[2007] - p[1952]).sort_values()
sel = list(gain.index[-10:][::-1]) + list(gain.index[:5])
fig, ax = plt.subplots(figsize=(9, 6.4))
ys = np.arange(len(sel))[::-1]
for yv, c in zip(ys, sel):
    bad = gain[c] < 15
    col = "#B91C1C" if bad else "#0B6E4F"
    ax.plot([p.loc[c, 1952], p.loc[c, 2007]], [yv, yv], color=col, lw=3, alpha=.45, solid_capstyle="round")
    ax.scatter(p.loc[c, 1952], yv, s=55, color="white", edgecolor=col, linewidth=2, zorder=5)
    ax.scatter(p.loc[c, 2007], yv, s=70, color=col, zorder=5)
    ax.text(max(p.loc[c, 1952], p.loc[c, 2007]) + 1.2, yv, f"{gain[c]:+.1f} yrs", va="center", fontsize=9, color=col, fontweight="bold")
ax.set_yticks(ys); ax.set_yticklabels(sel); ax.set_xlim(25, 88); ax.grid(axis="y", visible=False)
ax.axhline(4.5, color=MUTED, lw=.8, ls=":")
ax.text(87.5, 4.72, "Smallest gains (including losses)", fontsize=8.5, color="#B91C1C", style="italic", ha="right")
ax.text(26, 14.75, "Largest gains", fontsize=8.5, color="#0B6E4F", style="italic")
ax.set_ylim(-.8, 15.6)
ax.set_xlabel("Life expectancy at birth (years)")
ax.scatter([], [], s=55, color="white", edgecolor=MUTED, linewidth=2, label="1952"); ax.scatter([], [], s=70, color=MUTED, label="2007")
ax.legend(loc="lower right", frameon=False)
title(ax, "Some countries gained 30+ years; a few gained almost nothing", "Life expectancy in 1952 (hollow dot) and 2007 (solid dot)")
plt.savefig("fig3_dumbbell.png"); plt.close()
R["top_gainers"] = {c: round(float(gain[c]), 1) for c in gain.index[-5:][::-1]}
R["bottom_gainers"] = {c: round(float(gain[c]), 1) for c in gain.index[:5]}
R["mean_gain"] = round(float(gain.mean()), 1)
R["n_lost"] = int((gain < 0).sum())

# ---------- FIG 4: reversals ----------
hi = {"Zimbabwe": "#B91C1C", "Botswana": "#D55E00", "Swaziland": "#CC79A7", "South Africa": "#0072B2", "Rwanda": "#6A3D9A", "Cambodia": "#009E73"}
fig, ax = plt.subplots(figsize=(10, 5.8))
for c in p.index:
    if c not in hi: ax.plot(p.columns, p.loc[c], color="#D1D5DB", lw=.8, zorder=1)
for c, col in hi.items():
    ax.plot(p.columns, p.loc[c], color=col, lw=2.6, marker="o", ms=4, zorder=5)
ends = {"Zimbabwe": (2007.8, 43.2), "Botswana": (2007.8, 52.6), "Swaziland": (2007.8, 39.4), "South Africa": (2007.8, 48.8), "Rwanda": (2007.8, 46.9), "Cambodia": (2007.8, 59.7)}
for c, (xx, yy) in ends.items():
    ax.text(xx, yy, c, color=hi[c], fontsize=9.5, fontweight="bold", va="center")
ax.axvspan(1987, 2002, color="#FEF3C7", alpha=.55, zorder=0)
ax.text(1994.5, 84, "HIV/AIDS epidemic years (1987-2002)", ha="center", fontsize=8.5, color="#92400E")
ax.annotate("Rwanda 1992:\n23.6 years", xy=(1992, 23.6), xytext=(1996, 25), fontsize=9, color=hi["Rwanda"],
            arrowprops=dict(arrowstyle="->", color=hi["Rwanda"]))
ax.annotate("Cambodia 1977:\n31.2 years", xy=(1977, 31.22), xytext=(1962.5, 25.5), fontsize=9, color=hi["Cambodia"],
            arrowprops=dict(arrowstyle="->", color=hi["Cambodia"]))
ax.set_xlim(1951, 2016); ax.set_ylim(20, 88); ax.set_xticks(range(1952, 2008, 5)); ax.set_xticklabels(range(1952, 2008, 5), rotation=45, fontsize=8.5)
ax.set_ylabel("Life expectancy (years)")
title(ax, "But progress is not guaranteed", "Grey lines = all other countries. Coloured lines = countries where life expectancy collapsed.")
plt.savefig("fig4_reversals.png"); plt.close()
R["cambodia_1977"] = round(float(p.loc["Cambodia", 1977]), 1); R["rwanda_1992"] = round(float(p.loc["Rwanda", 1992]), 1)
R["rwanda_1987"] = round(float(p.loc["Rwanda", 1987]), 1)
R["zim"] = {int(y): round(float(p.loc["Zimbabwe", y]), 1) for y in [1987, 1992, 2002, 2007]}
R["bots"] = {int(y): round(float(p.loc["Botswana", y]), 1) for y in [1987, 1992, 2002]}
R["drop_8794"] = int(((p[2002] - p[1987]) < -5).sum())
R["drop_countries"] = [c for c in (p[2002] - p[1987]).sort_values().index[:7]]
R["drop_vals"] = {c: round(float((p[2002] - p[1987])[c]), 1) for c in (p[2002] - p[1987]).sort_values().index[:7]}
R["steps_down"] = int((p.diff(axis=1) < 0).sum().sum()); R["steps_total"] = int(p.diff(axis=1).notna().sum().sum())

# ---------- FIG 5: wealth vs health 2007, residuals ----------
d7 = df[df.year == 2007].copy()
slope, icpt = np.polyfit(d7.logGDP, d7.lifeExp, 1)
d7["pred"] = slope * d7.logGDP + icpt; d7["res"] = d7.lifeExp - d7.pred
r2 = np.corrcoef(d7.logGDP, d7.lifeExp)[0, 1] ** 2
fig, ax = plt.subplots(figsize=(10, 6))
xx = np.linspace(d7.logGDP.min(), d7.logGDP.max(), 100)
ax.plot(10 ** xx, slope * xx + icpt, color=INK, lw=1.6, ls="--", zorder=2)
ax.scatter(d7.gdpPercap, d7.lifeExp, s=48, c=d7.continent.map(PAL), alpha=.8, edgecolor="white", linewidth=.6, zorder=3)
over = d7.sort_values("res").tail(5); under = d7.sort_values("res").head(5)
OF = {"Vietnam": (-48, 26), "West Bank and Gaza": (16, 30), "Nicaragua": (-85, 10), "Comoros": (-34, 22), "Korea, Dem. Rep.": (16, 12),
      "Swaziland": (12, -12), "Angola": (14, 6), "Botswana": (18, -16), "South Africa": (-40, -28), "Equatorial Guinea": (18, 10)}
for _, r in over.iterrows():
    ax.annotate(r.country, (r.gdpPercap, r.lifeExp), xytext=OF[r.country], textcoords="offset points", fontsize=8.5, color="#0B6E4F", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#0B6E4F", lw=.7))
for _, r in under.iterrows():
    ax.annotate(r.country, (r.gdpPercap, r.lifeExp), xytext=OF[r.country], textcoords="offset points", fontsize=8.5, color="#B91C1C", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#B91C1C", lw=.7))
ax.set_xscale("log"); ax.set_xticks(xt); ax.set_xticklabels([f"${x:,}" for x in xt]); ax.set_xlim(200, 60000); ax.set_ylim(36, 86)
ax.set_xlabel("Income per person, 2007 (US$, log scale)"); ax.set_ylabel("Life expectancy (years)")
for k, v in PAL.items(): ax.scatter([], [], c=v, s=45, label=k)
ax.legend(loc="upper left", frameon=False, ncol=2)
title(ax, "Money buys health, but not automatically", "Dashed line = typical life expectancy for a given income. Green = healthier than income predicts; red = less healthy.")
plt.savefig("fig5_residuals.png"); plt.close()
R["r2_2007"] = round(float(r2), 2); R["slope"] = round(float(slope), 1)
R["over"] = {r.country: round(float(r.res), 1) for _, r in over.iloc[::-1].iterrows()}
R["under"] = {r.country: round(float(r.res), 1) for _, r in under.iterrows()}
R["double_income_years"] = round(float(slope * np.log10(2)), 1)

# ---------- FIG 6: continent journeys ----------
def agg(g): return pd.Series({"lifeavg": np.average(g.lifeExp, weights=g["pop"]), "gdpavg": np.average(g.gdpPercap, weights=g["pop"])})
cj = df.groupby(["continent", "year"]).apply(agg).reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
for c, g in cj.groupby("continent"):
    ax.plot(g.gdpavg, g.lifeavg, color=PAL[c], lw=2.4, alpha=.9, zorder=3)
    ax.scatter(g.gdpavg, g.lifeavg, s=26, color=PAL[c], zorder=4)
    ax.scatter(g.gdpavg.iloc[0], g.lifeavg.iloc[0], s=110, facecolor="white", edgecolor=PAL[c], linewidth=2.2, zorder=5)
    ax.scatter(g.gdpavg.iloc[-1], g.lifeavg.iloc[-1], s=130, color=PAL[c], edgecolor="white", linewidth=1.2, zorder=5)
lblpos = {"Africa": (8, -4), "Asia": (-48, 8), "Americas": (10, -4), "Europe": (-60, -2), "Oceania": (-62, 2)}
for c, g in cj.groupby("continent"):
    ax.annotate(c, (g.gdpavg.iloc[-1], g.lifeavg.iloc[-1]), xytext=lblpos[c], textcoords="offset points", color=PAL[c], fontsize=11, fontweight="bold")
ax.set_xscale("log"); ax.set_xticks([1000, 2000, 5000, 10000, 20000, 30000]); ax.set_xticklabels([f"${x:,}" for x in [1000, 2000, 5000, 10000, 20000, 30000]])
ax.set_xlim(600, 45000); ax.set_ylim(36, 84)
ax.set_xlabel("Average income per person (population-weighted, US$, log scale)"); ax.set_ylabel("Average life expectancy (years)")
ax.scatter([], [], s=110, facecolor="white", edgecolor=MUTED, linewidth=2.2, label="1952"); ax.scatter([], [], s=130, color=MUTED, label="2007")
ax.legend(loc="lower right", frameon=False)
title(ax, "Five continents, five different journeys", "Each path traces a continent from 1952 (hollow) to 2007 (solid), weighted by population.")
plt.savefig("fig6_paths.png"); plt.close()
cjp = cj.pivot(index="year", columns="continent", values="lifeavg"); cgp = cj.pivot(index="year", columns="continent", values="gdpavg")
R["cont_le"] = {c: [round(float(cjp.loc[1952, c]), 1), round(float(cjp.loc[2007, c]), 1)] for c in cjp.columns}
R["cont_gdp"] = {c: [round(float(cgp.loc[1952, c])), round(float(cgp.loc[2007, c]))] for c in cgp.columns}
R["africa_le_1987_2002"] = [round(float(cjp.loc[y, "Africa"]), 1) for y in [1987, 1992, 1997, 2002, 2007]]
# China / India
R["china"] = [round(float(p.loc["China", y]), 1) for y in [1952, 1957, 1962, 2007]]
R["china_gdp"] = [round(float(gp.loc["China", y])) for y in [1952, 2007]]
R["india_le"] = [round(float(p.loc["India", y]), 1) for y in [1952, 2007]]
R["growth_gdp_med"] = round(float(gp[2007].median() / gp[1952].median()), 1)
json.dump(R, open("results.json", "w"), indent=1)
print(json.dumps(R, indent=1))
